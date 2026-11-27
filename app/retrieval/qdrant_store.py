"""Local Qdrant REST vector-store adapter preserving SQLite chunk and document schemas."""

import json
import urllib.error
import urllib.request
import uuid

from app.core.config import settings
from app.security.network_guard import validate_local_endpoint


def _qdrant_request(
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: int = 30,
) -> dict:
    """Send an HTTP JSON request to the configured local Qdrant instance."""
    base_url = validate_local_endpoint(settings.QDRANT_URL, "Qdrant").rstrip("/")
    url = f"{base_url}{path}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None

    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _point_uuid(document_id: str, chunk_index: int) -> str:
    """Create a deterministic UUID string for a document chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}_{chunk_index}"))


def initialize_qdrant_collection(
    collection_name: str | None = None,
    vector_size: int = 384,
) -> dict:
    """Create the Qdrant collection if it does not already exist."""
    collection = collection_name or settings.QDRANT_COLLECTION

    try:
        return _qdrant_request("GET", f"/collections/{collection}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise

    return _qdrant_request(
        "PUT",
        f"/collections/{collection}",
        {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        },
    )


def store_qdrant_embeddings(
    document_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_name: str | None = None,
    filename: str | None = None,
) -> int:
    """Upsert document chunk vectors and metadata payloads into Qdrant."""
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")

    if not chunks:
        return 0

    collection = collection_name or settings.QDRANT_COLLECTION
    initialize_qdrant_collection(collection_name=collection, vector_size=len(embeddings[0]))

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        if len(embedding) != 384:
            raise ValueError("Expected a 384-dimensional embedding.")

        record_id = f"{document_id}_{chunk['chunk_index']}"
        payload = {
            "id": record_id,
            "document_id": document_id,
            "chunk_index": chunk["chunk_index"],
            "page_number": chunk.get("page_number"),
            "text": chunk["text"],
        }
        if filename:
            payload["filename"] = filename

        points.append({
            "id": _point_uuid(document_id, chunk["chunk_index"]),
            "vector": embedding,
            "payload": payload,
        })

    _qdrant_request(
        "PUT",
        f"/collections/{collection}/points?wait=true",
        {"points": points},
    )
    return len(points)


def get_qdrant_vectors(
    document_id: str | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """Scroll stored chunk records and vectors from Qdrant."""
    collection = collection_name or settings.QDRANT_COLLECTION
    records: list[dict] = []
    offset = None

    while True:
        body: dict = {
            "limit": 256,
            "with_payload": True,
            "with_vector": True,
        }
        if document_id is not None:
            body["filter"] = {
                "must": [
                    {
                        "key": "document_id",
                        "match": {"value": document_id},
                    }
                ]
            }
        if offset is not None:
            body["offset"] = offset

        response = _qdrant_request(
            "POST",
            f"/collections/{collection}/points/scroll",
            body,
        )
        result = response.get("result") or {}
        points = result.get("points") or []

        for point in points:
            payload = point.get("payload") or {}
            records.append({
                "id": payload.get("id", str(point.get("id"))),
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "page_number": payload.get("page_number"),
                "text": payload.get("text", ""),
                "embedding": point.get("vector") or [],
            })

        offset = result.get("next_page_offset")
        if offset is None or not points:
            break

    records.sort(key=lambda item: (item["document_id"] or "", item["chunk_index"] or 0))
    return records


def search_qdrant_vectors(
    query_embedding: list[float],
    top_k: int = 5,
    min_similarity: float = 0.30,
    document_id: str | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """Query Qdrant directly using cosine similarity and optional document filter."""
    collection = collection_name or settings.QDRANT_COLLECTION
    body: dict = {
        "vector": query_embedding,
        "limit": top_k,
        "with_payload": True,
        "score_threshold": min_similarity,
    }
    if document_id is not None:
        body["filter"] = {
            "must": [
                {
                    "key": "document_id",
                    "match": {"value": document_id},
                }
            ]
        }

    response = _qdrant_request(
        "POST",
        f"/collections/{collection}/points/search",
        body,
    )
    points = response.get("result") or []

    return [
        {
            "document_id": (point.get("payload") or {}).get("document_id"),
            "chunk_index": (point.get("payload") or {}).get("chunk_index"),
            "page_number": (point.get("payload") or {}).get("page_number"),
            "text": (point.get("payload") or {}).get("text", ""),
            "similarity_score": round(float(point.get("score", 0.0)), 6),
        }
        for point in points
    ]


def get_qdrant_vector_count(collection_name: str | None = None) -> int:
    """Return total point count in the Qdrant collection."""
    collection = collection_name or settings.QDRANT_COLLECTION
    response = _qdrant_request(
        "POST",
        f"/collections/{collection}/points/count",
        {"exact": True},
    )
    return int((response.get("result") or {}).get("count", 0))


def delete_qdrant_document_vectors(
    document_id: str,
    collection_name: str | None = None,
) -> int:
    """Delete all Qdrant points matching document_id and return deleted count."""
    collection = collection_name or settings.QDRANT_COLLECTION
    existing = get_qdrant_vectors(document_id=document_id, collection_name=collection)

    if not existing:
        return 0

    _qdrant_request(
        "POST",
        f"/collections/{collection}/points/delete?wait=true",
        {
            "filter": {
                "must": [
                    {
                        "key": "document_id",
                        "match": {"value": document_id},
                    }
                ]
            }
        },
    )
    return len(existing)
