"""
Fairness pass over the RAG eval: the agent sometimes answers correctly but emits
the option TEXT instead of the letter (e.g. "Answer: Abdominal Aortic Aneurysm"),
which the letter-based extractor scored as UNKNOWN -> wrong. That unfairly
penalises RAG. This script (no API calls) reloads the MedQA options, categorises
every UNKNOWN, recovers letters by matching stated answer text to option text, and
recomputes RAG accuracy honestly under several definitions.

Usage: python scripts/recover_unknowns.py <rag_jsonl>
"""

import os
import re
import sys
import json
import math
import difflib

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")

REFUSAL = ["couldn't find", "could not find", "no information", "not covered",
           "no relevant", "unable to find", "not available", "don't have",
           "outside the scope", "no data was retrieved"]
REWRITER = ["designed for rewriting", "not for answering", "rewriting clinical",
            "not a query for medical document"]


def wilson_ci(correct, total, z=1.96):
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (round(100 * p, 1), round(100 * (center - margin), 1), round(100 * (center + margin), 1))


def load_options_by_index():
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medqa", split="test")
    out = []
    for item in ds:
        d = item["data"]
        out.append({"options": d["Options"], "key": d["Correct Option"]})
    return out


def recover_letter(response, options):
    """Map a free-text final answer to an option letter, or None."""
    # candidate text after the last "Answer:"
    m = list(re.finditer(r"answer:\s*\**\s*(.+)", response, re.IGNORECASE))
    cands = []
    if m:
        cands.append(m[-1].group(1).strip().strip("*").strip())
    # also consider the whole last line
    last_line = response.strip().splitlines()[-1] if response.strip() else ""
    cands.append(last_line)

    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    opt_norm = {k: norm(v) for k, v in options.items()}

    for cand in cands:
        c = norm(cand)
        if not c:
            continue
        # exact / substring match
        for k, ov in opt_norm.items():
            if ov and (ov == c or ov in c or c in ov):
                return k
        # fuzzy best match
        best_k, best_r = None, 0.0
        for k, ov in opt_norm.items():
            if not ov:
                continue
            r = difflib.SequenceMatcher(None, c, ov).ratio()
            if r > best_r:
                best_r, best_k = r, k
        if best_r >= 0.85:
            return best_k
    return None


def main():
    rag_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not rag_path:
        import glob
        rag_path = max(glob.glob(os.path.join(EVAL_DIR, "*_rag.jsonl")), key=os.path.getmtime)
    rows = [json.loads(l) for l in open(rag_path, encoding="utf-8")]

    print(f"Loading MedQA options to align by index ...")
    opts = load_options_by_index()

    cats = {"recovered_correct": 0, "recovered_wrong": 0, "refusal": 0,
            "rewriter_reject": 0, "unrecoverable": 0}
    raw_correct = 0
    recovered_correct = 0   # original correct + recovered-correct
    answered = 0            # system committed to a choice (orig or recovered), for "committed accuracy"
    committed_correct = 0

    for r in rows:
        choice = r["agent_choice"]
        gold = r.get("correct_key")
        if choice not in ("UNKNOWN", "ERROR"):
            raw_correct += int(r.get("is_correct", False))
            recovered_correct += int(r.get("is_correct", False))
            answered += 1
            committed_correct += int(r.get("is_correct", False))
            continue

        # UNKNOWN/ERROR -> try recover
        idx = r["index"]
        options = opts[idx]["options"] if idx < len(opts) else {}
        resp = r.get("response", "") or ""
        low = resp.lower()
        rec = recover_letter(resp, options) if options else None

        if rec:
            ok = (rec == gold)
            cats["recovered_correct" if ok else "recovered_wrong"] += 1
            recovered_correct += int(ok)
            answered += 1
            committed_correct += int(ok)
        elif any(p in low for p in REWRITER):
            cats["rewriter_reject"] += 1
        elif any(p in low for p in REFUSAL):
            cats["refusal"] += 1
        else:
            cats["unrecoverable"] += 1

    n = len(rows)
    print("\n" + "=" * 64)
    print(f"RAG UNKNOWN FAIRNESS ANALYSIS  ({os.path.basename(rag_path)}, n={n})")
    print("=" * 64)
    print("\nUNKNOWN breakdown:")
    for k, v in cats.items():
        print(f"  {k:20s} {v}")

    raw = wilson_ci(raw_correct, n)
    recov = wilson_ci(recovered_correct, n)
    comm = wilson_ci(committed_correct, answered)
    print("\nRAG accuracy under three definitions:")
    print(f"  RAW (UNKNOWN=wrong, as reported): {raw[0]}%  (CI {raw[1]}-{raw[2]})  {raw_correct}/{n}")
    print(f"  RECOVERED (text-answers credited): {recov[0]}%  (CI {recov[1]}-{recov[2]})  {recovered_correct}/{n}")
    print(f"  COMMITTED-ONLY (excl. true refusals): {comm[0]}%  (CI {comm[1]}-{comm[2]})  {committed_correct}/{answered}")
    print("\n(no-RAG baseline for reference: 92.4%)")

    out = os.path.join(EVAL_DIR, "rag_fairness_analysis.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"RAG fairness analysis ({os.path.basename(rag_path)}, n={n})\n")
        f.write("UNKNOWN breakdown: " + json.dumps(cats) + "\n")
        f.write(f"RAW: {raw[0]}% ({raw_correct}/{n})\n")
        f.write(f"RECOVERED: {recov[0]}% ({recovered_correct}/{n})\n")
        f.write(f"COMMITTED-ONLY: {comm[0]}% ({committed_correct}/{answered})\n")
        f.write("no-RAG baseline: 92.4%\n")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
