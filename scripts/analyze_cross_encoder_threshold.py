
import json
from pathlib import Path

from sentence_transformers import CrossEncoder
from app.retrieval.vector_store import get_all_vectors

BASE_DIR = Path(__file__).resolve().parents[1]
CASES_FILE = BASE_DIR / "evaluation" / "retrieval_cases.json"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    model = CrossEncoder(MODEL_NAME, device="cpu")
    chunks = get_all_vectors()

    positive_scores = []
    negative_scores = []

    positive_hits = 0
    reciprocal_rank_sum = 0.0

    print("=" * 75)
    print("CROSS-ENCODER EVALUATION")
    print("=" * 75)

    for case in cases:
        query = case["query"]

        expected_document = case["expected_document_id"]
        expected_chunks = set(case["expected_chunk_indices"])

        candidates = chunks

        if expected_document:
            candidates = [
                chunk for chunk in chunks
                if chunk["document_id"] == expected_document
            ]

        pairs = [
            (query, chunk["text"])
            for chunk in candidates
        ]

        scores = model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        highest_score = float(ranked[0][1]) if ranked else float("-inf")

        if case["should_retrieve"]:
            relevant = [
                float(score)
                for chunk, score in ranked
                if chunk["chunk_index"] in expected_chunks
            ]

            if not relevant:
                print(f"\n{case['case_id']}: EXPECTED CHUNK MISSING")
                continue

            positive_scores.append(max(relevant))

            relevant_rank = next(
                rank
                for rank, (chunk, score) in enumerate(ranked, start=1)
                if chunk["chunk_index"] in expected_chunks
            )

            positive_hits += 1
            reciprocal_rank_sum += 1.0 / relevant_rank

            print(
                f"\nPOSITIVE | {max(relevant):.5f} | "
                f"Rank {relevant_rank} | {query}"
            )

        else:
            negative_scores.append(highest_score)

            print(
                f"\nNEGATIVE | {highest_score:.5f} | {query}"
            )

    print("\n" + "=" * 75)
    print("CROSS-ENCODER SCORE SUMMARY")
    print("=" * 75)

    print(f"Positive cases: {len(positive_scores)}")
    print(f"Negative cases: {len(negative_scores)}")

    if positive_scores:
        print(f"Positive minimum: {min(positive_scores):.5f}")
        print(f"Positive maximum: {max(positive_scores):.5f}")
        print(
            "Positive Hit@5: "
            f"{positive_hits / len(positive_scores):.4f}"
        )
        print(
            "Positive MRR@5: "
            f"{reciprocal_rank_sum / len(positive_scores):.4f}"
        )

    if negative_scores:
        print(f"Negative minimum: {min(negative_scores):.5f}")
        print(f"Negative maximum: {max(negative_scores):.5f}")


if __name__ == "__main__":
    main()
