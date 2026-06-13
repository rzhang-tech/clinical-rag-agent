"""
Simple-RAG ablation: isolate whether the AGENT orchestration or RETRIEVAL itself
hurts a weak open-weight model on MedQA.

Pipeline = ONE retrieval (same hybrid search + cross-encoder rerank the agent uses)
-> stuff top-k chunks into the prompt -> ONE LLM call. No query rewrite, no fan-out,
no orchestration loop, no compression. Compare against:
  - no-RAG (parametric only)   : 66.5% on Qwen 7B
  - agentic-RAG (full pipeline): 56.5% on Qwen 7B
If simple-RAG > agentic-RAG, the agent complexity is the culprit, not retrieval.

Run: LLM_PROVIDER=together python scripts/eval_simple_rag.py --sample 200 --seed 42
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import config
from langchain_core.messages import HumanMessage, SystemMessage
from scripts.evaluate import (load_medqa_questions, format_mcq_prompt,
                              extract_answer_regex, extract_answer_llm, wilson_ci, _retry)

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")

SIMPLE_SYS = SystemMessage(content=(
    "You are a medical exam assistant. Use the retrieved context below together with your "
    "own knowledge to answer the multiple-choice question. Reason briefly, then state your "
    "final answer on its own line as: **Answer: X**"
))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--run-id", type=str, default="qwen7b_simple")
    args = ap.parse_args()

    from db.vector_db_manager import VectorDbManager
    from db.cache_manager import CacheManager
    from rag_agent.tools import ToolFactory
    from core.rag_system import _create_llm

    cache = CacheManager()
    try:
        cache.connect()
    except Exception:
        pass
    vdb = VectorDbManager(cache=cache)
    vdb.create_collection(config.CHILD_COLLECTION)
    factory = ToolFactory(vdb.get_collection(config.CHILD_COLLECTION), cache=cache)
    llm = _create_llm()
    judge = _create_llm()

    questions = load_medqa_questions(sample=args.sample, seed=args.seed)
    jsonl = os.path.join(EVAL_DIR, f"{args.run_id}_simplerag.jsonl")
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            try:
                done.add(json.loads(line)["index"])
            except Exception:
                pass
    todo = [q for q in questions if q["index"] not in done]
    print(f"simple-RAG: {len(todo)} to run, {len(questions) - len(todo)} done")

    lock = __import__("threading").Lock()

    def run_one(q):
        keys = list(q["options"].keys())
        t0 = time.time()
        try:
            ctx = factory._search_child_chunks(q["question"], 5)  # single retrieval
            prompt = f"RETRIEVED CONTEXT:\n{ctx}\n\n{format_mcq_prompt(q['question'], q['options'])}"
            resp = _retry(lambda: llm.invoke([SIMPLE_SYS, HumanMessage(content=prompt)]).content)
        except Exception as e:
            return {"index": q["index"], "agent_choice": "ERROR", "is_correct": False,
                    "correct_key": q["correct_answer_key"], "elapsed_seconds": round(time.time()-t0, 1),
                    "response": f"ERROR: {e}"}
        choice = extract_answer_regex(resp, keys)
        if choice == "UNKNOWN":
            choice = extract_answer_llm(judge, resp, q["question"], q["options"], keys)
        return {"index": q["index"], "agent_choice": choice,
                "is_correct": choice == q["correct_answer_key"],
                "correct_key": q["correct_answer_key"],
                "elapsed_seconds": round(time.time()-t0, 1), "response": resp[:1000]}

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_one, q): q for q in todo}
        for fut in as_completed(futs):
            row = fut.result()
            with lock:
                with open(jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 20 == 0:
                print(f"  {completed}/{len(todo)} done")

    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    sc = [r for r in rows if r["agent_choice"] != "ERROR"]
    c = sum(1 for r in sc if r["is_correct"])
    err = len(rows) - len(sc)
    a, lo, hi = wilson_ci(c, len(sc))
    print("\n" + "=" * 50)
    print(f"SIMPLE-RAG (Qwen 7B, n={len(rows)})")
    print(f"  accuracy = {a}%  (95% CI {lo}-{hi})  {c}/{len(sc)}   ERROR={err}")
    print("  compare: no-RAG 66.5% | agentic-RAG 56.5%")
    print("=" * 50)


if __name__ == "__main__":
    main()
