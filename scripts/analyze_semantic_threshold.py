
import json
from pathlib import Path

from app.retrieval.semantic_search import semantic_search

BASE_DIR = Path(__file__).resolve().parents[1]
CASES_FILE = BASE_DIR / "evaluation" / "retrieval_cases.json"

TOP_K = 5


def main():
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    print("=" * 75)
    print("SEMANTIC SIMILARITY THRESHOLD ANALYSIS")
    print("=" * 75)

    positive_scores = []
    negative_scores = []

    for case in cases:
        results = semantic_search(
            query=case["query"],
            top_k=TOP_K,
            min_similarity=-1.0,
            document_id=case.get("expected_document_id"),
        )

        highest_score = (
            results[0]["similarity_score"] if results else 0.0
        )

        if case["should_retrieve"]:
            positive_scores.append(highest_score)
            category = "POSITIVE"
        else:
            negative_scores.append(highest_score)
            category = "NEGATIVE"

        print(
            f"{category:8} | "
            f"{highest_score:.5f} | "
            f"{case['query']}"
        )

    print("\n" + "=" * 75)
    print("SCORE SUMMARY")
    print("=" * 75)

    print(
        f"Positive minimum: "
        f"{min(positive_scores):.5f}"
    )
    print(
        f"Positive maximum: "
        f"{max(positive_scores):.5f}"
    )
    print(
        f"Negative minimum: "
        f"{min(negative_scores):.5f}"
    )
    print(
        f"Negative maximum: "
        f"{max(negative_scores):.5f}"
    )


if __name__ == "__main__":
    main()
