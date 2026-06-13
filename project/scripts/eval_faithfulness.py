"""
Faithfulness / groundedness eval (RAGAS-style) for the RAG agent.

Answers the question "does it make things up?" — i.e. are the agent's answers
grounded in the context it actually retrieved, or fabricated. The main eval does
not capture retrieved context (the agent compresses/deletes tool messages), so
here we wrap the retrieval tools to capture every chunk the agent saw, then have
an LLM judge score, per question:
  - faithfulness  = fraction of the answer's clinical claims supported by context
  - context_relevance = how relevant the retrieved context is to the question

Sampled (default 120) because a full re-run is ~6h; faithfulness is a rate and
120 gives a usable CI. Sequential so per-question context capture is clean.
Checkpointed to a JSONL so it is resumable.

Usage: python scripts/eval_faithfulness.py --sample 120 --seed 42
"""

import os
import re
import sys
import json
import math
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_core.messages import HumanMessage, SystemMessage
import config
from scripts.evaluate import load_medqa_questions, format_mcq_prompt, extract_answer_regex, wilson_ci

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")

# Capture buffer — populated by the wrapped retrieval tools (sequential => safe).
_CAPTURED = []


def build_system_with_capture():
    from db.vector_db_manager import VectorDbManager
    from db.cache_manager import CacheManager
    from db.postgres_manager import PostgresManager
    from rag_agent.tools import ToolFactory
    from rag_agent.graph import create_agent_graph
    from core.rag_system import _create_llm

    pg = PostgresManager()
    try:
        pg.connect()
    except Exception:
        pass
    cache = CacheManager()
    try:
        cache.connect()
    except Exception:
        pass

    vdb = VectorDbManager(cache=cache)
    vdb.create_collection(config.CHILD_COLLECTION)
    collection = vdb.get_collection(config.CHILD_COLLECTION)

    llm = _create_llm()
    factory = ToolFactory(collection, cache=cache)

    import functools
    orig_search = factory._search_child_chunks
    orig_parent = factory._retrieve_parent_chunks

    @functools.wraps(orig_search)  # copy docstring + annotations so tool() is happy
    def wrapped_search(query, limit):
        out = orig_search(query, limit)
        if out and not out.startswith(("NO_RELEVANT", "RETRIEVAL_ERROR")):
            _CAPTURED.append(out)
        return out

    @functools.wraps(orig_parent)
    def wrapped_parent(parent_id):
        out = orig_parent(parent_id)
        if out and not out.startswith(("NO_PARENT", "PARENT_RETRIEVAL_ERROR")):
            _CAPTURED.append(out)
        return out

    factory._search_child_chunks = wrapped_search
    factory._retrieve_parent_chunks = wrapped_parent

    tools = factory.create_tools()
    graph = create_agent_graph(llm, tools)
    judge = _create_llm()
    return graph, judge


def _json_from(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


FAITH_SYS = SystemMessage(content=(
    "You are a strict faithfulness judge for a medical RAG system. You are given the "
    "CONTEXT the system retrieved and the ANSWER it produced. Judge ONLY whether the "
    "answer's clinical claims are supported by the CONTEXT — do NOT use your own medical "
    "knowledge. Output strict JSON only: "
    '{\"claims_total\": <int>, \"claims_supported\": <int>, \"grounded\": <true|false>}. '
    "grounded=true only if every material clinical claim in the answer is supported by the context."
))

REL_SYS = SystemMessage(content=(
    "Rate how relevant the retrieved CONTEXT is to answering the QUESTION, from 0.0 (irrelevant) "
    'to 1.0 (directly relevant). Output strict JSON only: {\"relevance\": <float>}.'
))


def judge_faithfulness(judge, context, answer):
    if not context.strip():
        return None
    msg = HumanMessage(content=f"CONTEXT:\n{context[:12000]}\n\nANSWER:\n{answer[:4000]}")
    try:
        d = _json_from(judge.invoke([FAITH_SYS, msg]).content)
        if d and d.get("claims_total", 0) > 0:
            return {"faithfulness": round(d["claims_supported"] / d["claims_total"], 3),
                    "grounded": bool(d.get("grounded")),
                    "claims_total": d["claims_total"], "claims_supported": d["claims_supported"]}
    except Exception as e:
        return {"error": str(e)[:100]}
    return None


def judge_relevance(judge, question, context):
    if not context.strip():
        return None
    msg = HumanMessage(content=f"QUESTION:\n{question[:2000]}\n\nCONTEXT:\n{context[:12000]}")
    try:
        d = _json_from(judge.invoke([REL_SYS, msg]).content)
        if d and "relevance" in d:
            return round(float(d["relevance"]), 3)
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)
    jsonl = os.path.join(EVAL_DIR, f"{args.run_id}_faithfulness.jsonl")
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            try:
                done.add(json.loads(line)["index"])
            except Exception:
                pass

    questions = load_medqa_questions(sample=args.sample, seed=args.seed)
    todo = [q for q in questions if q["index"] not in done]
    print(f"Faithfulness eval: {len(todo)} to run ({len(done)} resumed). Building system...")

    import uuid
    graph, judge = build_system_with_capture()

    for i, q in enumerate(todo, 1):
        _CAPTURED.clear()
        prompt = format_mcq_prompt(q["question"], q["options"])
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": config.GRAPH_RECURSION_LIMIT}
        t0 = time.time()
        try:
            res = graph.invoke({"messages": [HumanMessage(content=prompt)]}, cfg)
            answer = res["messages"][-1].content
        except Exception as e:
            answer = f"ERROR: {e}"
        context = "\n\n---\n\n".join(dict.fromkeys(_CAPTURED))  # unique, preserve order

        faith = judge_faithfulness(judge, context, answer)
        rel = judge_relevance(judge, q["question"], context)
        choice = extract_answer_regex(answer, list(q["options"].keys()))

        row = {
            "index": q["index"], "question": q["question"][:300],
            "correct_key": q["correct_answer_key"], "agent_choice": choice,
            "is_correct": choice == q["correct_answer_key"],
            "has_context": bool(context.strip()), "n_retrievals": len(_CAPTURED),
            "faithfulness": faith, "context_relevance": rel,
            "elapsed_seconds": round(time.time() - t0, 1),
            "answer": answer[:1500],
        }
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        fz = faith.get("faithfulness") if isinstance(faith, dict) else None
        print(f"[{i}/{len(todo)}] ctx={'Y' if row['has_context'] else 'N'} "
              f"faith={fz} rel={rel} pick={choice}/{q['correct_answer_key']} {row['elapsed_seconds']}s")

    summarize(jsonl)


def summarize(jsonl):
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    with_ctx = [r for r in rows if r["has_context"]]
    faiths = [r["faithfulness"]["faithfulness"] for r in with_ctx
              if isinstance(r["faithfulness"], dict) and "faithfulness" in r["faithfulness"]]
    grounded = [r for r in with_ctx if isinstance(r["faithfulness"], dict) and r["faithfulness"].get("grounded")]
    rels = [r["context_relevance"] for r in with_ctx if isinstance(r["context_relevance"], (int, float))]

    no_ctx = len(rows) - len(with_ctx)
    lines = []
    lines.append("=" * 60)
    lines.append(f"FAITHFULNESS / GROUNDEDNESS  (n={len(rows)}, with-context={len(with_ctx)}, no-context={no_ctx})")
    lines.append("=" * 60)
    if faiths:
        mean_f = round(sum(faiths) / len(faiths), 3)
        gr = wilson_ci(len(grounded), len(with_ctx))
        lines.append(f"  mean faithfulness        = {mean_f}  (1.0 = every claim supported)")
        lines.append(f"  fully-grounded answers   = {gr[0]}%  (CI {gr[1]}-{gr[2]})  {len(grounded)}/{len(with_ctx)}")
    if rels:
        lines.append(f"  mean context relevance   = {round(sum(rels)/len(rels),3)}")
    lines.append(f"  answered WITHOUT retrieving any context: {no_ctx}/{len(rows)} "
                 f"({round(100*no_ctx/len(rows),0)}%)")
    report = "\n".join(lines)
    print("\n" + report)
    out = os.path.join(EVAL_DIR, "faithfulness_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
