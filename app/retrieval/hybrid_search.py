
from app.retrieval.semantic_search import semantic_search
from app.retrieval.lexical_search import lexical_search


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 50,
    rrf_k: int = 60,
    document_id: str | None = None,
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k <= 0 or candidate_k <= 0 or rrf_k <= 0:
        raise ValueError("Search parameters must be positive.")

    semantic_results = semantic_search(
        query=query,
        top_k=candidate_k,
        min_similarity=-1.0,
        document_id=document_id,
    )

    lexical_results = lexical_search(
        query=query,
        top_k=candidate_k,
        document_id=document_id,
    )

    fused_results = {}

    def get_key(result):
        return (
            result["document_id"],
            result["chunk_index"],
        )

    def initialize_result(result):
        key = get_key(result)

        if key not in fused_results:
            fused_results[key] = {
                "document_id": result["document_id"],
                "chunk_index": result["chunk_index"],
                "page_number": result["page_number"],
                "text": result["text"],
                "rrf_score": 0.0,
                "semantic_rank": None,
                "lexical_rank": None,
                "semantic_score": None,
                "bm25_score": None,
            }

        return fused_results[key]

    # Dense retrieval contributions.
    for rank, result in enumerate(semantic_results, start=1):

        item = initialize_result(result)

        item["rrf_score"] += 1.0 / (rrf_k + rank)
        item["semantic_rank"] = rank
        item["semantic_score"] = result["similarity_score"]

    # Lexical retrieval contributions.
    for rank, result in enumerate(lexical_results, start=1):

        item = initialize_result(result)

        item["rrf_score"] += 1.0 / (rrf_k + rank)
        item["lexical_rank"] = rank
        item["bm25_score"] = result["bm25_score"]

    results = list(fused_results.values())

    # Deterministic sorting:
    # 1. Higher RRF score
    # 2. Better lexical rank
    # 3. Better semantic rank
    # 4. Lower chunk index
    results.sort(
        key=lambda item: (
            -item["rrf_score"],
            item["lexical_rank"] or float("inf"),
            item["semantic_rank"] or float("inf"),
            item["chunk_index"],
        )
    )

    return results[:top_k]
