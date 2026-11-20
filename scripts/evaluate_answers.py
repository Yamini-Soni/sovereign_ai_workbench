
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from app.retrieval.semantic_search import semantic_search

from scripts.test_llm_answer import (
    MODEL,
    TOP_K,
    MIN_SIMILARITY,
    NOT_FOUND,
    generate_grounded_answer,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

OUT_FILE = BASE_DIR / "evaluation" / "answer_results.json"

EXPERIMENT_NAME = "Local RAG with Evidence Verification"
PROMPT_VERSION = "v2-detailed"
VERIFIER_VERSION = "password_length_v1"

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Evaluation dataset
# --------------------------------------------------

CASES = [
    {
        "id": "Q01",
        "question": "How many days of annual leave do employees receive?",
        "answerable": True,
        "expected_any": ["20"],
        "forbidden": [],
    },
    {
        "id": "Q02",
        "question": "Who approves leave requests?",
        "answerable": True,
        "expected_any": ["line manager", "manager"],
        "forbidden": [],
    },
    {
        "id": "Q03",
        "question": "Where must employees submit leave requests?",
        "answerable": True,
        "expected_any": ["hr portal"],
        "forbidden": [],
    },
    {
        "id": "Q04",
        "question": "How often must passwords be changed?",
        "answerable": True,
        "expected_any": ["90 days", "90"],
        "forbidden": [],
    },
    {
        "id": "Q05",
        "question": "What is the minimum password length?",
        "answerable": True,
        "expected_any": ["12"],
        "forbidden": ["not explicitly stated"],
    },
    {
        "id": "Q06",
        "question": "Are employees allowed to share their passwords?",
        "answerable": True,
        "expected_any": [
            "not allowed",
            "not permitted",
            "prohibited",
            "cannot share",
        ],
        "forbidden": [],
    },
    {
        "id": "Q07",
        "question": "How long are backups retained?",
        "answerable": True,
        "expected_any": ["30 days", "30"],
        "forbidden": [],
    },
    {
        "id": "Q08",
        "question": "At what time are servers backed up?",
        "answerable": True,
        "expected_any": ["2 am", "2am", "2:00"],
        "forbidden": [],
    },
    {
        "id": "Q09",
        "question": "How frequently are backup restoration tests performed?",
        "answerable": True,
        "expected_any": ["quarter"],
        "forbidden": [],
    },
    {
        "id": "Q10",
        "question": "Within how much time must security incidents be reported?",
        "answerable": True,
        "expected_any": ["1 hour", "one hour"],
        "forbidden": [],
    },
    {
        "id": "Q11",
        "question": "What is the CEO's annual salary?",
        "answerable": False,
        "expected_any": [],
        "forbidden": [],
    },
    {
        "id": "Q12",
        "question": "Which programming language is used for the backend?",
        "answerable": False,
        "expected_any": [],
        "forbidden": [],
    },
    {
        "id": "Q13",
        "question": "What health insurance benefits do employees receive?",
        "answerable": False,
        "expected_any": [],
        "forbidden": [],
    },
    {
        "id": "Q14",
        "question": "How many employees work at NexaTech?",
        "answerable": False,
        "expected_any": [],
        "forbidden": [],
    },
    {
        "id": "Q15",
        "question": "Where is the company's head office located?",
        "answerable": False,
        "expected_any": [],
        "forbidden": [],
    },
]


# --------------------------------------------------
# Evaluation functions
# --------------------------------------------------

def is_refusal(answer: str) -> bool:

    return "could not find" in answer.lower()


def evaluate_answer(case: dict, answer: str) -> bool:

    answer_lower = answer.lower()

    # An unanswerable question passes only if
    # the system correctly refuses to answer.
    if not case["answerable"]:
        return is_refusal(answer)

    # An answerable question must not be refused.
    if is_refusal(answer):
        return False

    # At least one expected answer pattern must appear.
    has_expected_content = any(
        keyword.lower() in answer_lower
        for keyword in case["expected_any"]
    )

    if not has_expected_content:
        return False

    # Reject known contradictory phrases.
    has_contradiction = any(
        phrase.lower() in answer_lower
        for phrase in case["forbidden"]
    )

    if has_contradiction:
        return False

    return True


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    print("=" * 70)
    print("LOCAL RAG ANSWER EVALUATION")
    print("=" * 70)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Model: {MODEL}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Verifier version: {VERIFIER_VERSION}")
    print(f"Top K: {TOP_K}")
    print(f"Minimum similarity: {MIN_SIMILARITY}")
    print(f"Total questions: {len(CASES)}")

    print("=" * 70)

    results = []

    for case in CASES:

        question = case["question"]

        print("\n" + "-" * 70)
        print(f"{case['id']} | {question}")
        print("-" * 70)

        start_time = time.perf_counter()

        chunks = []
        answer = None
        method = None
        evidence = None
        error = None

        try:

            # Step 1: Retrieve relevant document chunks.
            chunks = semantic_search(
                query=question,
                top_k=TOP_K,
                min_similarity=MIN_SIMILARITY,
                document_id=None,
            )

            # Step 2: Generate answer using the
            # evidence-verification and local-LLM pipeline.
            if chunks:

                answer, method, evidence = generate_grounded_answer(
                    question=question,
                    chunks=chunks,
                )

            else:

                answer = NOT_FOUND
                method = "no_retrieval"

        except Exception as exception:

            error = str(exception)

        elapsed = time.perf_counter() - start_time

        # Step 3: Evaluate the answer.
        if error is not None:

            passed = False
            status = "ERROR"

        else:

            passed = evaluate_answer(case, answer)
            status = "PASS" if passed else "FAIL"

        # Step 4: Display individual results.
        print(f"Status: {status}")
        print(f"Answer: {answer if answer is not None else error}")
        print(f"Answer method: {method}")
        print(f"Time: {elapsed:.2f} seconds")

        if evidence:

            print("Verified evidence:")
            print(f"  {evidence['evidence']}")
            print(f"  Page: {evidence['page_number']}")

        if chunks:

            print("Retrieved sources:")

            for chunk in chunks:

                print(
                    f"  Chunk {chunk['chunk_index']} "
                    f"| Page {chunk['page_number']} "
                    f"| Similarity {chunk['similarity_score']}"
                )

        else:

            print("Retrieved sources: None")

        # Step 5: Save individual result.
        results.append({

            "id": case["id"],
            "question": question,
            "answerable": case["answerable"],

            "answer": answer,
            "answer_method": method,

            "verified_evidence": (
                evidence["evidence"] if evidence else None
            ),

            "evidence_page": (
                evidence["page_number"] if evidence else None
            ),

            "passed": passed,
            "status": status,

            "time_seconds": round(elapsed, 3),

            "retrieved_chunks": [
                {
                    "chunk_index": chunk["chunk_index"],
                    "page_number": chunk["page_number"],
                    "similarity": chunk["similarity_score"],
                }
                for chunk in chunks
            ],

            "error": error,
        })

    # --------------------------------------------------
    # Calculate evaluation metrics
    # --------------------------------------------------

    answerable_results = [
        result for result in results
        if result["answerable"]
    ]

    unanswerable_results = [
        result for result in results
        if not result["answerable"]
    ]

    answerable_correct = sum(
        result["passed"] for result in answerable_results
    )

    unanswerable_correct = sum(
        result["passed"] for result in unanswerable_results
    )

    total_passed = sum(
        result["passed"] for result in results
    )

    answerable_accuracy = (
        answerable_correct / len(answerable_results) * 100
        if answerable_results else 0
    )

    refusal_accuracy = (
        unanswerable_correct / len(unanswerable_results) * 100
        if unanswerable_results else 0
    )

    overall_accuracy = (
        total_passed / len(results) * 100
        if results else 0
    )

    response_times = [
        result["time_seconds"]
        for result in results
        if result["error"] is None
    ]

    median_time = (
        statistics.median(response_times)
        if response_times else 0
    )

    # Count answer methods.
    method_counts = {}

    for result in results:

        method = result["answer_method"] or "error"

        method_counts[method] = (
            method_counts.get(method, 0) + 1
        )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Answerable questions correct: "
        f"{answerable_correct}/{len(answerable_results)} "
        f"({answerable_accuracy:.1f}%)"
    )

    print(
        f"Unanswerable questions refused correctly: "
        f"{unanswerable_correct}/{len(unanswerable_results)} "
        f"({refusal_accuracy:.1f}%)"
    )

    print(
        f"Overall keyword-based pass rate: "
        f"{total_passed}/{len(results)} "
        f"({overall_accuracy:.1f}%)"
    )

    print(f"Median response time: {median_time:.2f} seconds")

    print("\nAnswer method distribution:")

    for method, count in method_counts.items():

        print(f"  {method}: {count}")

    print("\nImportant:")
    print(
        "The keyword-based evaluator is a preliminary metric. "
        "Manual review is still required to identify incorrect, "
        "incomplete, or contradictory answers."
    )

    # --------------------------------------------------
    # Save experiment results
    # --------------------------------------------------

    report = {

        "experiment": EXPERIMENT_NAME,

        "date": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),

        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "verifier_version": VERIFIER_VERSION,

        "top_k": TOP_K,
        "minimum_similarity": MIN_SIMILARITY,

        "total_questions": len(results),

        "answerable_correct": answerable_correct,
        "answerable_total": len(answerable_results),

        "answerable_accuracy_percent": round(
            answerable_accuracy, 2
        ),

        "unanswerable_refused_correctly": unanswerable_correct,
        "unanswerable_total": len(unanswerable_results),

        "refusal_accuracy_percent": round(
            refusal_accuracy, 2
        ),

        "overall_passed": total_passed,

        "overall_accuracy_percent": round(
            overall_accuracy, 2
        ),

        "median_response_time_seconds": round(
            median_time, 3
        ),

        "answer_method_distribution": method_counts,

        "results": results,
    }

    OUT_FILE.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nResults saved to:")
    print(OUT_FILE)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
