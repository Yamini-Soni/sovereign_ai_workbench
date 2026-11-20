
import json
from pathlib import Path

from app.retrieval.lexical_search import lexical_search

BASE_DIR = Path(__file__).resolve().parents[1]
CASES_FILE = BASE_DIR / "evaluation" / "retrieval_cases.json"


def main():
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    print("=" * 70)
    print("BM25 SCORE ANALYSIS")
    print("=" * 70)

    for case in cases:
        results = lexical_search(
            query=case["query"],
            top_k=100000,
            document_id=case.get("expected_document_id"),
        )

        print(f"\n{case['case_id']}: {case['query']}")

        if not results:
            print("No matching chunks.")
            continue

        if case["should_retrieve"]:
            expected_chunks = set(case["expected_chunk_indices"])

            matching = [
                result for result in results
                if result["chunk_index"] in expected_chunks
            ]

            if matching:
                print(
                    "Expected chunk BM25 score:",
                    matching[0]["bm25_score"],
                )
            else:
                print("Expected chunk not found.")

        print(
            "Highest BM25 score:",
            results[0]["bm25_score"],
        )


if __name__ == "__main__":
    main()
