"""
Combine the latest no-RAG and RAG eval JSONL checkpoints into one ablation report.

Reads the newest *_norag.jsonl and *_rag.jsonl in eval_results/, computes accuracy
with a 95% Wilson confidence interval for each, and the retrieval-effect delta.
Writes ablation_summary_<timestamp>.txt and prints to stdout.
"""

import os
import re
import glob
import json
import math
from datetime import datetime

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")


def wilson_ci(correct, total, z=1.96):
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (round(100 * p, 1), round(100 * (center - margin), 1), round(100 * (center + margin), 1))


def newest(pattern):
    files = glob.glob(os.path.join(EVAL_DIR, pattern))
    return max(files, key=os.path.getmtime) if files else None


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def summarize(path):
    rows = load_rows(path)
    scored = [r for r in rows if r.get("agent_choice") != "ERROR"]
    errors = len(rows) - len(scored)
    correct = sum(1 for r in scored if r.get("is_correct"))
    acc, lo, hi = wilson_ci(correct, len(scored))
    return {"file": os.path.basename(path), "n": len(rows), "scored": len(scored),
            "errors": errors, "correct": correct, "acc": acc, "lo": lo, "hi": hi}


def main():
    norag_path = newest("*_norag.jsonl")
    rag_path = newest("*_rag.jsonl")

    lines = []
    lines.append("=" * 64)
    lines.append("ABLATION REPORT — Gemini 2.5 Flash on MedQA (RAG vs no-RAG)")
    lines.append(f"generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 64)

    s_norag = summarize(norag_path) if norag_path else None
    s_rag = summarize(rag_path) if rag_path else None

    for tag, s in (("NO-RAG (parametric only)", s_norag), ("RAG (retrieval + agent)", s_rag)):
        lines.append("")
        if not s:
            lines.append(f"[{tag}]  no data file found")
            continue
        lines.append(f"[{tag}]  file={s['file']}")
        lines.append(f"  n={s['n']}  scored={s['scored']}  errors={s['errors']}")
        lines.append(f"  accuracy = {s['acc']}%   (95% CI {s['lo']}–{s['hi']})   correct={s['correct']}/{s['scored']}")

    if s_norag and s_rag:
        delta = round(s_rag["acc"] - s_norag["acc"], 1)
        sign = "+" if delta >= 0 else ""
        lines.append("")
        lines.append("-" * 64)
        lines.append("RETRIEVAL EFFECT")
        lines.append(f"  RAG {s_rag['acc']}%  −  no-RAG {s_norag['acc']}%  =  Δ {sign}{delta} pts")
        overlap = not (s_rag["lo"] > s_norag["hi"] or s_norag["lo"] > s_rag["hi"])
        lines.append(f"  CIs {'OVERLAP (gap not significant at this n)' if overlap else 'DISJOINT (gap is significant)'}")
        lines.append("-" * 64)

    report = "\n".join(lines)
    print(report)

    out = os.path.join(EVAL_DIR, f"ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
