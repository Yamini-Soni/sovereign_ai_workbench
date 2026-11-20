
from sentence_transformers import CrossEncoder

from app.retrieval.vector_store import get_all_vectors


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TEST_QUERIES = [
    "How many days of annual leave do employees receive?",
    "Who approves leave requests?",
    "How often must passwords be changed?",
    "How long are backups retained?",
    "How many employees work at NexaTech?",
    "What is the CEO's annual salary?",
]


def main():
    print("Loading cross-encoder model...")

    model = CrossEncoder(
        MODEL_NAME,
        device="cpu",
    )

    chunks = get_all_vectors()

    if not chunks:
        print("No stored chunks found.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    for query in TEST_QUERIES:
        print("\n" + "=" * 70)
        print(f"QUESTION: {query}")
        print("=" * 70)

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        scores = model.predict(pairs)

        ranked_results = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        for rank, (chunk, score) in enumerate(
            ranked_results,
            start=1,
        ):
            print(f"\nRank: {rank}")
            print(f"Chunk: {chunk['chunk_index']}")
            print(f"Cross-encoder score: {float(score):.5f}")
            print(f"Text:\n{chunk['text'][:500]}")


if __name__ == "__main__":
    main()
