
import json
from pathlib import Path

from app.retrieval.semantic_search import semantic_search

BASE_DIR = Path(__file__).resolve().parents[1]
CASES_FILE = BASE_DIR / "evaluation" / "retrieval_cases.json"

TOP_K = 5
MIN_SIMILARITY = 0.30


def main():
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    positive_cases = 0
    negative_cases = 0
    hits = 0
    reciprocal_rank_sum = 0.0
    correctly_rejected = 0

    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for case in cases:
        print(f"\nCase: {case['case_id']}")
        print(f"Question: {case['query']}")

        should_retrieve = case["should_retrieve"]

        results = semantic_search(
            query=case["query"],
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
            document_id=case.get("expected_document_id"),
        )

        if should_retrieve:
            positive_cases += 1

            expected_document = case["expected_document_id"]
            expected_chunks = set(case["expected_chunk_indices"])

            relevant_rank = None

            for rank, result in enumerate(results, start=1):
                if (
                    result["document_id"] == expected_document
                    and result["chunk_index"] in expected_chunks
                ):
                    relevant_rank = rank
                    break

            if relevant_rank is not None:
                hits += 1
                reciprocal_rank_sum += 1.0 / relevant_rank
                print(f"Result: HIT at rank {relevant_rank}")

            else:
                print("Result: MISS")

                # Diagnostic search: inspect all chunks in the
                # expected document without the production threshold.
                diagnostic_results = semantic_search(
                    query=case["query"],
                    top_k=100000,
                    min_similarity=-1.0,
                    document_id=expected_document,
                )

                diagnostic_rank = None
                diagnostic_score = None

                for rank, result in enumerate(
                    diagnostic_results, start=1
                ):
                    if result["chunk_index"] in expected_chunks:
                        diagnostic_rank = rank
                        diagnostic_score = result["similarity_score"]
                        break

                if diagnostic_rank is not None:
                    print(
                        "Expected chunk diagnostic rank:",
                        diagnostic_rank,
                    )
                    print(
                        "Expected chunk similarity:",
                        diagnostic_score,
                    )
                else:
                    print(
                        "Expected chunk was not found in diagnostic search."
                    )

                print("Top 5 diagnostic candidates:")

                for result in diagnostic_results[:5]:
                    print(
                        f"  Chunk {result['chunk_index']} | "
                        f"Page {result['page_number']} | "
                        f"Score {result['similarity_score']}"
                    )

        else:
            negative_cases += 1

            if not results:
                correctly_rejected += 1
                print("Result: CORRECTLY REJECTED")
            else:
                print("Result: FALSE ACCEPTANCE")
                print(
                    "Highest similarity:",
                    results[0]["similarity_score"],
                )

    hit_at_k = (
        hits / positive_cases if positive_cases else 0.0
    )

    mrr_at_k = (
        reciprocal_rank_sum / positive_cases
        if positive_cases else 0.0
    )

    negative_rejection_rate = (
        correctly_rejected / negative_cases
        if negative_cases else 0.0
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Positive cases: {positive_cases}")
    print(f"Negative cases: {negative_cases}")
    print(f"Hit@{TOP_K}: {hit_at_k:.4f}")
    print(f"MRR@{TOP_K}: {mrr_at_k:.4f}")
    print(
        "Negative rejection rate: "
        f"{negative_rejection_rate:.4f}"
    )


if __name__ == "__main__":
    main()
