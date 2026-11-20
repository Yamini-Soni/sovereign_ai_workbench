
import math
import re

from app.retrieval.vector_store import get_all_vectors


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_score(
    query_tokens: list[str],
    document_tokens: list[str],
    document_frequencies: dict,
    total_documents: int,
    average_document_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:

    if not document_tokens or average_document_length == 0:
        return 0.0

    document_length = len(document_tokens)
    term_frequencies = {}

    for token in document_tokens:
        term_frequencies[token] = (
            term_frequencies.get(token, 0) + 1
        )

    score = 0.0

    for token in set(query_tokens):
        frequency = term_frequencies.get(token, 0)

        if frequency == 0:
            continue

        document_frequency = document_frequencies.get(token, 0)

        inverse_document_frequency = math.log(
            1 + (
                total_documents - document_frequency + 0.5
            ) / (
                document_frequency + 0.5
            )
        )

        numerator = frequency * (k1 + 1)

        denominator = frequency + k1 * (
            1 - b + b * document_length / average_document_length
        )

        score += (
            inverse_document_frequency
            * numerator
            / denominator
        )

    return score


def lexical_search(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    stored_vectors = get_all_vectors()

    if document_id:
        stored_vectors = [
            item for item in stored_vectors
            if item["document_id"] == document_id
        ]

    if not stored_vectors:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    tokenized_documents = [
        tokenize(item["text"])
        for item in stored_vectors
    ]

    total_documents = len(tokenized_documents)

    document_frequencies = {}

    for tokens in tokenized_documents:
        for token in set(tokens):
            document_frequencies[token] = (
                document_frequencies.get(token, 0) + 1
            )

    average_document_length = (
        sum(len(tokens) for tokens in tokenized_documents)
        / total_documents
    )

    results = []

    for item, document_tokens in zip(
        stored_vectors,
        tokenized_documents,
    ):

        score = bm25_score(
            query_tokens=query_tokens,
            document_tokens=document_tokens,
            document_frequencies=document_frequencies,
            total_documents=total_documents,
            average_document_length=average_document_length,
        )

        if score <= 0:
            continue

        results.append({
            "document_id": item["document_id"],
            "chunk_index": item["chunk_index"],
            "page_number": item["page_number"],
            "text": item["text"],
            "bm25_score": round(score, 6),
        })

    results.sort(
        key=lambda result: result["bm25_score"],
        reverse=True,
    )

    return results[:top_k]
