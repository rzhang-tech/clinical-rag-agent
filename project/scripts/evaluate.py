"""
Clinical RAG Agent — Evaluation (v2)

Measures what actually matters for a retrieval system and reports numbers that
survive scrutiny:

  1. Answer accuracy on MedQA (USMLE-style MCQ), with a 95% Wilson confidence
     interval so a reader knows the uncertainty given the sample size.
  2. RAG vs. no-RAG ablation — the same model answering the same questions with
     retrieval ON (full agent) and OFF (LLM parametric knowledge only). The gap
     is the only honest evidence that the retrieval stack adds value, since a
     strong LLM can answer many MCQs from memory.
  3. Robust answer extraction: fast regex first, LLM-judge fallback when the
     regex is ambiguous (the old version silently scored un-parseable answers as
     wrong, deflating accuracy).
  4. Resumable runs: every question is appended to a JSONL checkpoint as it
     finishes, so a crash 8 hours into a full-set run loses nothing.

What was DELETED from v1 and why:
  - The keyword-based in-KB / out-KB classifier. It labelled pediatrics,
    psychiatry, obstetrics and gynecology as "out of knowledge base" — but those
    textbooks ARE in the KB (Nelson, DSM-5, Williams, Novak). Every metric built
    on that split was methodologically invalid. Boundary/refusal evaluation needs
    a properly constructed out-of-domain probe set; that is future work, not a
    keyword guess. See `--ood-note`.

Usage:
    python scripts/evaluate.py --mode both --sample 20          # smoke test
    python scripts/evaluate.py --mode both --full               # full test set (~1273 q)
    python scripts/evaluate.py --mode both --sample 300 --seeds 3   # 3-seed subset, mean±std
    python scripts/evaluate.py --mode rag --full --concurrency 6
    python scripts/evaluate.py --resume <run_id>                # continue an interrupted run
"""

import sys
import os
import re
import json
import math
import time
import argparse
import threading
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")


# --------------------------------------------------------------------------- #
# Dataset                                                                       #
# --------------------------------------------------------------------------- #

def load_medqa_questions(sample: int = 0, seed: int = 42):
    """Load MedQA test questions.

    sample=0 -> the full test split (the whole population; a single run is
    enough and a Wilson CI captures the residual uncertainty).
    sample>0 -> a deterministic random subset of size `sample` for that seed.
    """
    print(f"Loading MedQA test split (sample={sample or 'FULL'}, seed={seed})...")
    ds = load_dataset("openlifescienceai/medqa", split="test")

    entries = []
    for item in ds:
        data = item["data"]
        entries.append({
            "question": data["Question"],
            "options": data["Options"],
            "correct_answer_key": data["Correct Option"],
            "correct_answer_text": data["Correct Answer"],
        })

    if sample and sample < len(entries):
        import random
        rng = random.Random(seed)
        entries = rng.sample(entries, sample)

    # Stable index so resume/checkpoint identity is deterministic per (seed, sample).
    for i, e in enumerate(entries):
        e["index"] = i
    print(f"  {len(entries)} questions loaded")
    return entries


def format_mcq_prompt(question: str, options: dict) -> str:
    options_text = "\n".join(f"  {key}. {val}" for key, val in sorted(options.items()))
    return (
        f"{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Select the single best answer and explain your reasoning. "
        f"State your final answer on its own line as: **Answer: X**"
    )


# --------------------------------------------------------------------------- #
# Answer extraction: regex fast-path + LLM-judge fallback                       #
# --------------------------------------------------------------------------- #

_EXTRACT_PATTERNS = [
    r"\*\*Answer:\s*([A-Z])\*\*",
    r"Answer:\s*([A-Z])\b",
    r"(?:correct|best)\s+answer\s+is\s+\(?([A-Z])\)?\b",
    r"(?:option|choice)\s+\(?([A-Z])\)?\s+is\s+correct",
]


def extract_answer_regex(response: str, valid_keys: list) -> str:
    for pattern in _EXTRACT_PATTERNS:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        for m in reversed(matches):
            choice = m.upper()
            if choice in valid_keys:
                return choice
    return "UNKNOWN"


def extract_answer_llm(judge_llm, response: str, question: str, options: dict, valid_keys: list) -> str:
    """LLM-judge fallback: read the agent's prose and report which option it chose.
    This judges *extraction only* (which letter did the answer settle on), not
    correctness — so it cannot inflate accuracy, only recover un-parseable picks.
    """
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    sys_msg = SystemMessage(content=(
        "You are an answer-extraction tool. Given a multiple-choice question and a "
        "candidate response, output ONLY the single letter of the option the response "
        "concludes is correct. If the response makes no clear choice, output NONE. "
        "Output exactly one token: a letter or NONE."
    ))
    human = HumanMessage(content=(
        f"QUESTION:\n{question}\n\nOPTIONS:\n{options_text}\n\n"
        f"RESPONSE:\n{response}\n\nWhich option did the response choose?"
    ))
    try:
        out = judge_llm.invoke([sys_msg, human]).content.strip().upper()
        m = re.search(r"[A-Z]", out)
        if m and m.group(0) in valid_keys:
            return m.group(0)
    except Exception:
        pass
    return "UNKNOWN"


_SOURCE_HINT = re.compile(r"(parent id|file name|source[s]?:|\.md\b|textbook)", re.IGNORECASE)


def has_source_attribution(response: str) -> bool:
    """Heuristic: does the answer cite where it came from? Weak signal, reported
    separately from accuracy and never folded into the headline number."""
    return bool(_SOURCE_HINT.search(response))


# --------------------------------------------------------------------------- #
# Statistics: Wilson score interval (closed-form, no scipy)                      #
# --------------------------------------------------------------------------- #

def wilson_ci(correct: int, total: int, z: float = 1.96):
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (round(100 * p, 1), round(100 * (center - margin), 1), round(100 * (center + margin), 1))


# --------------------------------------------------------------------------- #
# Runners                                                                        #
# --------------------------------------------------------------------------- #

def _retry(fn, attempts: int = 4, base: float = 2.0):
    """Exponential backoff — survives Gemini rate-limit / transient 5xx so a
    long run does not die on a single 429."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            if any(t in msg for t in ("429", "rate", "quota", "resource", "503", "unavailable",
                                       "timeout", "422", "grammar", "failed to compile")):
                time.sleep(base ** i)
                continue
            raise
    raise last


def run_rag(rag_system, judge_llm, q: dict) -> dict:
    import uuid
    from langchain_core.messages import HumanMessage as HM

    prompt = format_mcq_prompt(q["question"], q["options"])
    valid_keys = list(q["options"].keys())
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": rag_system.recursion_limit}

    start = time.time()
    try:
        result = _retry(lambda: rag_system.agent_graph.invoke(
            {"messages": [HM(content=prompt)]}, cfg))
        response = result["messages"][-1].content
    except Exception as exc:  # noqa: BLE001
        return {**_base_row(q, "rag"), "agent_choice": "ERROR",
                "elapsed_seconds": round(time.time() - start, 1),
                "response": f"ERROR: {exc}"}

    chosen = extract_answer_regex(response, valid_keys)
    if chosen == "UNKNOWN":
        chosen = extract_answer_llm(judge_llm, response, q["question"], q["options"], valid_keys)

    return {**_base_row(q, "rag"),
            "agent_choice": chosen,
            "is_correct": chosen == q["correct_answer_key"],
            "has_sources": has_source_attribution(response),
            "elapsed_seconds": round(time.time() - start, 1),
            "response": response}


_NORAG_SYS = SystemMessage(content=(
    "You are a medical exam assistant. Answer the multiple-choice question using "
    "your own knowledge. Reason briefly, then state your final answer on its own "
    "line as: **Answer: X**"
))


def run_norag(norag_llm, judge_llm, q: dict) -> dict:
    prompt = format_mcq_prompt(q["question"], q["options"])
    valid_keys = list(q["options"].keys())
    start = time.time()
    try:
        response = _retry(lambda: norag_llm.invoke([_NORAG_SYS, HumanMessage(content=prompt)]).content)
    except Exception as exc:  # noqa: BLE001
        return {**_base_row(q, "norag"), "agent_choice": "ERROR",
                "elapsed_seconds": round(time.time() - start, 1),
                "response": f"ERROR: {exc}"}

    chosen = extract_answer_regex(response, valid_keys)
    if chosen == "UNKNOWN":
        chosen = extract_answer_llm(judge_llm, response, q["question"], q["options"], valid_keys)

    return {**_base_row(q, "norag"),
            "agent_choice": chosen,
            "is_correct": chosen == q["correct_answer_key"],
            "has_sources": False,
            "elapsed_seconds": round(time.time() - start, 1),
            "response": response}


def _base_row(q: dict, mode: str) -> dict:
    return {
        "index": q["index"],
        "mode": mode,
        "question": q["question"],
        "correct_answer": f"{q['correct_answer_key']}. {q['correct_answer_text']}",
        "correct_key": q["correct_answer_key"],
        "agent_choice": "UNKNOWN",
        "is_correct": False,
        "has_sources": False,
        "elapsed_seconds": 0.0,
        "response": "",
    }


# --------------------------------------------------------------------------- #
# Checkpointed orchestration                                                     #
# --------------------------------------------------------------------------- #

_write_lock = threading.Lock()


def _load_done(jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["mode"], r["index"]))
            except Exception:
                continue
    return done


def _append(jsonl_path: str, row: dict):
    with _write_lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_condition(mode: str, questions, runner, jsonl_path: str, concurrency: int, limit: int = 0):
    done = _load_done(jsonl_path)
    todo = [q for q in questions if (mode, q["index"]) not in done]
    already = len(questions) - len(todo)
    if limit and len(todo) > limit:
        todo = todo[:limit]  # run only this batch; checkpoint remembers the rest
    print(f"\n[{mode}] running {len(todo)} this batch, {already} already done, "
          f"{len(questions) - already - len(todo)} remaining after this batch")

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(runner, q): q for q in todo}
        for fut in as_completed(futures):
            row = fut.result()
            _append(jsonl_path, row)
            results.append(row)
            completed += 1
            status = "OK " if row["is_correct"] else ("ERR" if row["agent_choice"] == "ERROR" else "x  ")
            print(f"  [{mode} {completed}/{len(todo)}] {status} "
                  f"pick={row['agent_choice']} gold={row.get('correct_key','?')} "
                  f"{row['elapsed_seconds']}s")

    # Merge with anything already on disk so the summary reflects the full set.
    all_rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_rows.append(json.loads(line))
            except Exception:
                continue
    return [r for r in all_rows if r["mode"] == mode]


def summarize(mode: str, rows: list) -> dict:
    scored = [r for r in rows if r["agent_choice"] != "ERROR"]
    errors = len(rows) - len(scored)
    correct = sum(1 for r in scored if r["is_correct"])
    acc, lo, hi = wilson_ci(correct, len(scored))
    avg_t = round(sum(r["elapsed_seconds"] for r in rows) / len(rows), 1) if rows else 0
    src = sum(1 for r in scored if r.get("has_sources"))
    return {
        "mode": mode, "n": len(rows), "scored": len(scored), "errors": errors,
        "correct": correct, "accuracy": acc, "ci95_low": lo, "ci95_high": hi,
        "source_rate": round(100 * src / len(scored), 1) if scored else 0,
        "avg_time_seconds": avg_t,
    }


def print_summary(summaries: dict):
    print("\n" + "=" * 64)
    print("EVALUATION SUMMARY")
    print("=" * 64)
    for s in summaries.values():
        print(f"\n[{s['mode'].upper()}]  n={s['n']}  (scored={s['scored']}, errors={s['errors']})")
        print(f"  Accuracy:        {s['accuracy']}%  (95% CI {s['ci95_low']}–{s['ci95_high']})")
        print(f"  Correct:         {s['correct']}/{s['scored']}")
        if s["mode"] == "rag":
            print(f"  Source-cited:    {s['source_rate']}%")
        print(f"  Avg time/q:      {s['avg_time_seconds']}s")

    if "rag" in summaries and "norag" in summaries:
        r, n = summaries["rag"], summaries["norag"]
        delta = round(r["accuracy"] - n["accuracy"], 1)
        print("\n" + "-" * 64)
        print("ABLATION — retrieval contribution")
        print(f"  RAG ON:  {r['accuracy']}%   RAG OFF: {n['accuracy']}%")
        sign = "+" if delta >= 0 else ""
        print(f"  Δ (retrieval effect): {sign}{delta} pts")
        print("  (Honest headline: RAG lifts MedQA accuracy by the Δ above; "
              "CIs show whether the gap is significant at this n.)")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Main                                                                           #
# --------------------------------------------------------------------------- #

def run_once(modes, questions, run_id: str, concurrency: int, seed_tag: str = "", limit: int = 0):
    os.makedirs(EVAL_DIR, exist_ok=True)
    summaries = {}

    # Lazy init — only build what each mode needs.
    rag_system = judge_llm = norag_llm = None
    if "rag" in modes or "norag" in modes:
        from core.rag_system import _create_llm
        judge_llm = _create_llm()
        norag_llm = _create_llm()
    if "rag" in modes:
        from core.rag_system import RAGSystem
        print("\nInitializing RAG system (Qdrant + parent store + agent graph)...")
        rag_system = RAGSystem()
        rag_system.initialize()

    for mode in modes:
        jsonl = os.path.join(EVAL_DIR, f"{run_id}{seed_tag}_{mode}.jsonl")
        if mode == "rag":
            runner = lambda q: run_rag(rag_system, judge_llm, q)  # noqa: E731
            # Agent graph + shared cross-encoder: cap concurrency (docker Qdrant server handles it).
            cc = min(concurrency, 8)
        else:
            runner = lambda q: run_norag(norag_llm, judge_llm, q)  # noqa: E731
            cc = concurrency
        rows = run_condition(mode, questions, runner, jsonl, cc, limit)
        summaries[mode] = summarize(mode, rows)

    return summaries


def main():
    ap = argparse.ArgumentParser(description="Clinical RAG Agent evaluation (v2)")
    ap.add_argument("--mode", choices=["rag", "norag", "both"], default="both")
    ap.add_argument("--full", action="store_true", help="Run the entire MedQA test split")
    ap.add_argument("--sample", type=int, default=20, help="Subset size (ignored if --full)")
    ap.add_argument("--seeds", type=int, default=1, help="Number of seeds for subset runs (mean±std)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Run only N not-yet-done questions this invocation (batching)")
    ap.add_argument("--resume", type=str, default="", help="Resume an existing run_id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    modes = ["rag", "norag"] if args.mode == "both" else [args.mode]
    sample = 0 if args.full else args.sample
    run_id = args.resume or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        qs = load_medqa_questions(sample=sample or 5, seed=42)
        for q in qs[:5]:
            print(f"  [{q['index']}] {q['question'][:80]}...  -> {q['correct_answer_key']}")
        return

    # Full set or single-seed: one pass. Multi-seed only makes sense for subsets.
    seeds = [42] if (args.full or args.seeds <= 1) else [42 + i for i in range(args.seeds)]

    per_seed = []
    for si, seed in enumerate(seeds):
        seed_tag = "" if len(seeds) == 1 else f"_s{seed}"
        if len(seeds) > 1:
            print(f"\n########## SEED {seed} ({si+1}/{len(seeds)}) ##########")
        questions = load_medqa_questions(sample=sample, seed=seed)
        summaries = run_once(modes, questions, run_id, args.concurrency, seed_tag, args.limit)
        print_summary(summaries)
        per_seed.append(summaries)

    # Aggregate across seeds (subset multi-seed only).
    if len(per_seed) > 1:
        print("\n" + "#" * 64)
        print(f"MULTI-SEED AGGREGATE  ({len(per_seed)} seeds, n={sample} each)")
        print("#" * 64)
        for mode in modes:
            accs = [ps[mode]["accuracy"] for ps in per_seed]
            mean = round(statistics.mean(accs), 1)
            std = round(statistics.pstdev(accs), 1) if len(accs) > 1 else 0.0
            print(f"  [{mode}] {mean}% ± {std}  (seeds: {accs})")

    # Persist a compact summary JSON next to the JSONL checkpoints.
    summary_path = os.path.join(EVAL_DIR, f"{run_id}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "full": args.full, "sample": sample,
                   "seeds": seeds, "per_seed": per_seed}, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {summary_path}")
    print("Per-question JSONL checkpoints in", EVAL_DIR)


if __name__ == "__main__":
    main()
