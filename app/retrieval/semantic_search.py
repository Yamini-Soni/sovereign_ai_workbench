
import math

from app.retrieval.embedding_service import generate_embeddings
from app.retrieval.vector_store import get_all_vectors


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def semantic_search(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.30,
    document_id: str | None = None,
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not -1 <= min_similarity <= 1:
        raise ValueError("Invalid similarity threshold.")

    # Generate query embedding.
    query_embedding = generate_embeddings([query.strip()])[0]

    # Retrieve stored document vectors.
    stored_vectors = get_all_vectors()

    # Optionally restrict search to one document.
    if document_id:
        stored_vectors = [
            item for item in stored_vectors
            if item["document_id"] == document_id
        ]

    results = []

    for item in stored_vectors:

        score = cosine_similarity(
            query_embedding,
            item["embedding"],
        )

        # Reject low-similarity results.
        if score < min_similarity:
            continue

        results.append({
            "document_id": item["document_id"],
            "chunk_index": item["chunk_index"],
            "page_number": item["page_number"],
            "text": item["text"],
            "similarity_score": round(score, 6),
        })

    results.sort(
        key=lambda result: result["similarity_score"],
        reverse=True,
    )

    return results[:top_k]
