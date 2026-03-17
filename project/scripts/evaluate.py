"""
Clinical RAG Agent Evaluation Script

Uses MedQA (USMLE-style multiple choice) questions to evaluate:
1. Answer Accuracy - does the agent pick the correct answer?
2. Retrieval Quality - does the agent find relevant documents?
3. Boundary Awareness - does the agent refuse when knowledge is missing?

Usage:
    python scripts/evaluate.py                    # Run with defaults (20 questions)
    python scripts/evaluate.py --num-questions 50 # Run 50 questions
    python scripts/evaluate.py --dry-run          # Preview questions without querying agent
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from datasets import load_dataset
from langchain_core.messages import HumanMessage
from core.rag_system import RAGSystem


# --- Topics covered by our knowledge base ---
IN_KB_TOPICS = {
    "anatomy", "cardiovascular", "heart", "blood", "vessel", "artery", "vein",
    "internal medicine", "diabetes", "hypertension", "hepatitis", "renal",
    "liver", "kidney", "lung", "pulmonary", "cardiac", "gastrointestinal",
    "pathology", "neoplasm", "tumor", "cancer", "inflammation", "necrosis",
    "pharmacology", "drug", "medication", "antibiotic", "receptor", "inhibitor",
    "clinical", "diagnosis", "treatment", "therapy", "symptom", "sign",
}

# Topics NOT in our knowledge base
OUT_KB_TOPICS = {
    "pediatric", "child", "infant", "neonatal", "newborn",
    "psychiatric", "depression", "schizophrenia", "bipolar", "anxiety disorder",
    "obstetric", "pregnancy", "labor", "delivery", "prenatal",
    "gynecology", "cervical", "ovarian", "uterine",
}


def classify_question(question_text: str, options: dict) -> str:
    """Classify whether a question is likely in-KB or out-KB."""
    text = (question_text + " " + " ".join(options.values())).lower()

    out_score = sum(1 for kw in OUT_KB_TOPICS if kw in text)
    in_score = sum(1 for kw in IN_KB_TOPICS if kw in text)

    if out_score > in_score:
        return "out_kb"
    return "in_kb"


def load_medqa_questions(num_in_kb: int = 15, num_out_kb: int = 5, seed: int = 42):
    """Load a balanced set of MedQA questions."""
    print("Loading MedQA dataset from HuggingFace...")
    ds = load_dataset("openlifescienceai/medqa", split="test")

    in_kb_questions = []
    out_kb_questions = []

    for item in ds:
        data = item["data"]
        question = data["Question"]
        options = data["Options"]
        answer_key = data["Correct Option"]
        answer_text = data["Correct Answer"]

        category = classify_question(question, options)

        entry = {
            "question": question,
            "options": options,
            "correct_answer_key": answer_key,
            "correct_answer_text": answer_text,
            "category": category,
        }

        if category == "in_kb" and len(in_kb_questions) < num_in_kb:
            in_kb_questions.append(entry)
        elif category == "out_kb" and len(out_kb_questions) < num_out_kb:
            out_kb_questions.append(entry)

        if len(in_kb_questions) >= num_in_kb and len(out_kb_questions) >= num_out_kb:
            break

    all_questions = in_kb_questions + out_kb_questions
    print(f"  Selected {len(in_kb_questions)} in-KB + {len(out_kb_questions)} out-KB = {len(all_questions)} questions")
    return all_questions


def format_mcq_prompt(question: str, options: dict) -> str:
    """Format a multiple-choice question for the agent."""
    options_text = "\n".join(f"  {key}. {val}" for key, val in sorted(options.items()))
    return (
        f"{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Select the single best answer and explain your reasoning based on the retrieved evidence. "
        f"State your final answer as: **Answer: X**"
    )


def extract_answer_choice(response: str, valid_keys: list) -> str:
    """Extract the chosen answer letter from agent response."""
    import re

    # Ordered from most explicit to least explicit
    patterns = [
        r"\*\*Answer:\s*([A-Z])\*\*",                    # **Answer: B**
        r"Answer:\s*([A-Z])\b",                           # Answer: B
        r"(?:correct|best)\s+answer\s+is\s+\(?([A-Z])\)?\b",  # correct answer is B / (B)
        r"(?:option|choice)\s+\(?([A-Z])\)?\s+is\s+correct",  # option B is correct
        r"\b([A-Z])\)\s",                                 # B) ...
        r"^\s*\(?([A-Z])\)?\.\s",                         # A. ... at start of line
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            choice = matches[-1].upper()
            if choice in valid_keys:
                return choice

    # Fallback: use LLM-as-judge via simple heuristic
    # Count how many times each option letter appears in bold or standalone
    option_counts = {}
    for key in valid_keys:
        # Count prominent mentions: bold, parenthesized, or "Option X"
        count = len(re.findall(rf"\*\*{key}\*\*|\({key}\)|Option\s+{key}\b", response))
        if count > 0:
            option_counts[key] = count

    if option_counts:
        return max(option_counts, key=option_counts.get)

    return "UNKNOWN"


def judge_boundary_awareness(response: str) -> bool:
    """Check if the agent appropriately refused to answer (for out-KB questions)."""
    refusal_phrases = [
        "couldn't find",
        "could not find",
        "no relevant",
        "not available",
        "no information",
        "outside",
        "not covered",
        "insufficient",
        "unable to find",
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in refusal_phrases)


def run_evaluation(num_in_kb: int = 15, num_out_kb: int = 5, dry_run: bool = False):
    """Run the full evaluation pipeline."""

    # Load questions
    questions = load_medqa_questions(num_in_kb=num_in_kb, num_out_kb=num_out_kb)

    if dry_run:
        print("\n[DRY RUN] Preview of selected questions:\n")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. [{q['category']}] {q['question'][:80]}...")
            print(f"     Correct: {q['correct_answer_key']}. {q['correct_answer_text']}")
        print(f"\nTotal: {len(questions)} questions")
        return

    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag_system = RAGSystem()
    rag_system.initialize()

    # Run evaluation
    results = []
    total = len(questions)

    print(f"\nRunning evaluation on {total} questions...\n")

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{total}] [{q['category']}] {q['question'][:70]}...")

        prompt = format_mcq_prompt(q["question"], q["options"])

        start_time = time.time()
        try:
            # Reset thread for each question (independent evaluation)
            rag_system.reset_thread()

            result = rag_system.agent_graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                rag_system.get_config()
            )
            response = result["messages"][-1].content
            elapsed = time.time() - start_time

            # Evaluate
            valid_keys = list(q["options"].keys())
            chosen = extract_answer_choice(response, valid_keys)
            is_correct = chosen == q["correct_answer_key"]
            is_refusal = judge_boundary_awareness(response)
            has_sources = "sources:" in response.lower() or "source:" in response.lower()

            results.append({
                "index": i,
                "category": q["category"],
                "question": q["question"],
                "correct_answer": f"{q['correct_answer_key']}. {q['correct_answer_text']}",
                "agent_choice": chosen,
                "is_correct": is_correct,
                "is_refusal": is_refusal,
                "has_sources": has_sources,
                "elapsed_seconds": round(elapsed, 1),
                "response": response,
            })

            status = "✅" if is_correct else ("🔇" if is_refusal else "❌")
            print(f"  {status} Agent: {chosen} | Correct: {q['correct_answer_key']} | {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  💥 Error: {str(e)[:80]} | {elapsed:.1f}s")
            results.append({
                "index": i,
                "category": q["category"],
                "question": q["question"],
                "correct_answer": f"{q['correct_answer_key']}. {q['correct_answer_text']}",
                "agent_choice": "ERROR",
                "is_correct": False,
                "is_refusal": False,
                "has_sources": False,
                "elapsed_seconds": round(elapsed, 1),
                "response": f"ERROR: {str(e)}",
            })

    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    in_kb = [r for r in results if r["category"] == "in_kb"]
    out_kb = [r for r in results if r["category"] == "out_kb"]

    # In-KB metrics
    in_kb_correct = sum(1 for r in in_kb if r["is_correct"])
    in_kb_with_sources = sum(1 for r in in_kb if r["has_sources"])
    in_kb_accuracy = in_kb_correct / len(in_kb) * 100 if in_kb else 0

    print(f"\n📚 In-KB Questions ({len(in_kb)} total):")
    print(f"  Accuracy:         {in_kb_correct}/{len(in_kb)} ({in_kb_accuracy:.0f}%)")
    print(f"  With Sources:     {in_kb_with_sources}/{len(in_kb)}")

    # Out-KB metrics
    out_kb_refusals = sum(1 for r in out_kb if r["is_refusal"])
    out_kb_correct = sum(1 for r in out_kb if r["is_correct"])
    boundary_rate = out_kb_refusals / len(out_kb) * 100 if out_kb else 0

    print(f"\n🚫 Out-KB Questions ({len(out_kb)} total):")
    print(f"  Refusal Rate:     {out_kb_refusals}/{len(out_kb)} ({boundary_rate:.0f}%)")
    print(f"  Correct Despite:  {out_kb_correct}/{len(out_kb)} (answered correctly using general knowledge)")

    # Overall
    total_correct = in_kb_correct + out_kb_correct
    overall_accuracy = total_correct / len(results) * 100 if results else 0
    avg_time = sum(r["elapsed_seconds"] for r in results) / len(results) if results else 0

    print(f"\n📊 Overall:")
    print(f"  Accuracy:         {total_correct}/{len(results)} ({overall_accuracy:.0f}%)")
    print(f"  Avg Time/Question: {avg_time:.1f}s")

    # Save detailed results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_{timestamp}.json")

    report = {
        "timestamp": timestamp,
        "config": {
            "num_in_kb": num_in_kb,
            "num_out_kb": num_out_kb,
            "score_threshold": 0.5,
        },
        "metrics": {
            "in_kb_accuracy": round(in_kb_accuracy, 1),
            "in_kb_source_rate": round(in_kb_with_sources / len(in_kb) * 100, 1) if in_kb else 0,
            "out_kb_refusal_rate": round(boundary_rate, 1),
            "overall_accuracy": round(overall_accuracy, 1),
            "avg_time_seconds": round(avg_time, 1),
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Clinical RAG Agent")
    parser.add_argument("--num-in-kb", type=int, default=15, help="Number of in-KB questions")
    parser.add_argument("--num-out-kb", type=int, default=5, help="Number of out-KB questions")
    parser.add_argument("--dry-run", action="store_true", help="Preview questions without running agent")
    args = parser.parse_args()

    run_evaluation(num_in_kb=args.num_in_kb, num_out_kb=args.num_out_kb, dry_run=args.dry_run)
