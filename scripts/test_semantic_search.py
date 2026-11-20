
from app.retrieval.semantic_search import semantic_search


def main():

    print("=" * 55)
    print("SOVEREIGN AI WORKBENCH - SEMANTIC SEARCH")
    print("=" * 55)

    query = input("\nEnter your question: ").strip()

    try:
        results = semantic_search(
            query=query,
            top_k=5,
            min_similarity=0.30,
        )

        if not results:
            print("\nNo sufficiently relevant chunks found.")
            print(
                "The document collection may not contain "
                "information relevant to your question."
            )
            return

        print("\nTop matching document chunks:\n")

        for rank, result in enumerate(results, start=1):

            print("-" * 55)
            print(f"Rank: {rank}")
            print(f"Document ID: {result['document_id']}")
            print(f"Page: {result['page_number']}")
            print(f"Chunk index: {result['chunk_index']}")
            print(
                f"Similarity: {result['similarity_score']:.4f}"
            )
            print("\nContent:")
            print(result["text"])

        print("-" * 55)

    except Exception as error:
        print(f"\nSearch failed: {error}")


if __name__ == "__main__":
    main()
