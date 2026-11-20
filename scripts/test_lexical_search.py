
from app.retrieval.lexical_search import lexical_search

DOCUMENT_ID = "7c0b7f6c-ec8a-4078-8da6-383b8a21936f"

queries = [
    "In which year was the Canada Company formed by John Galt?",
    "What year did John Galt form the Canada Company?",
    "Identify the year associated with Galt's formation of the Canada Company.",
]

for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    results = lexical_search(
        query=query,
        top_k=5,
        document_id=DOCUMENT_ID,
    )

    if not results:
        print("No matching chunks found.")
        continue

    for rank, result in enumerate(results, start=1):

        print(
            f"\nRank: {rank} | "
            f"Chunk: {result['chunk_index']} | "
            f"Page: {result['page_number']} | "
            f"BM25: {result['bm25_score']}"
        )

        print(result["text"][:300])
