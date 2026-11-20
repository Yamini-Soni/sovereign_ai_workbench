
from app.retrieval.hybrid_search import hybrid_search

DOCUMENT_ID = "7c0b7f6c-ec8a-4078-8da6-383b8a21936f"

queries = [
    "In which year was the Canada Company formed by John Galt?",
    "What year did John Galt form the Canada Company?",
    "Identify the year associated with Galt's formation of the Canada Company.",
]

for query in queries:

    print("\n" + "=" * 75)
    print("QUERY:", query)
    print("=" * 75)

    results = hybrid_search(
        query=query,
        top_k=5,
        candidate_k=50,
        document_id=DOCUMENT_ID,
    )

    for rank, result in enumerate(results, start=1):

        print(
            f"\nRank: {rank} | "
            f"Chunk: {result['chunk_index']} | "
            f"Page: {result['page_number']} | "
            f"RRF: {result['rrf_score']:.6f}"
        )

        print("Semantic rank:", result["semantic_rank"])
        print("Lexical rank:", result["lexical_rank"])
        print("Semantic score:", result["semantic_score"])
        print("BM25 score:", result["bm25_score"])
        print("Text:", result["text"][:250])
