"""Migrate stored vectors and metadata from SQLite to local Qdrant with parity verification."""

import json
import sqlite3
from pathlib import Path

from app.core.config import BASE_DIR, settings
from app.retrieval.qdrant_store import (
    get_qdrant_vector_count,
    get_qdrant_vectors,
    initialize_qdrant_collection,
    search_qdrant_vectors,
    store_qdrant_embeddings,
)
from app.retrieval.semantic_search import cosine_similarity
from app.storage.metadata_store import get_all_documents


SQLITE_DB_PATH = BASE_DIR / "data" / "vector_store.db"


def load_sqlite_vectors(db_path: Path = SQLITE_DB_PATH) -> list[dict]:
    """Read all stored chunk records and embeddings directly from SQLite."""
    if not db_path.exists():
        return []

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM document_vectors ORDER BY document_id, chunk_index"
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "text": row["text"],
            "embedding": json.loads(row["embedding"]),
        }
        for row in rows
    ]


def migrate_sqlite_to_qdrant(
    db_path: Path = SQLITE_DB_PATH,
    collection_name: str | None = None,
    verify_parity: bool = True,
) -> dict:
    """
    Migrate all SQLite vectors into Qdrant while preserving IDs, embeddings, and metadata.

    Returns a migration summary including count and retrieval parity checks.
    """
    collection = collection_name or settings.QDRANT_COLLECTION
    records = load_sqlite_vectors(db_path)

    if not records:
        return {
            "status": "no_records",
            "sqlite_count": 0,
            "qdrant_count": 0,
            "parity_verified": True,
        }

    filename_lookup = {
        doc["document_id"]: doc.get("filename")
        for doc in get_all_documents()
        if doc.get("document_id")
    }

    initialize_qdrant_collection(
        collection_name=collection,
        vector_size=len(records[0]["embedding"]),
    )

    by_doc: dict[str, list[dict]] = {}
    for record in records:
        by_doc.setdefault(record["document_id"], []).append(record)

    migrated = 0
    for document_id, doc_records in by_doc.items():
        chunks = [
            {
                "chunk_index": item["chunk_index"],
                "page_number": item["page_number"],
                "text": item["text"],
            }
            for item in doc_records
        ]
        embeddings = [item["embedding"] for item in doc_records]
        migrated += store_qdrant_embeddings(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection,
            filename=filename_lookup.get(document_id),
        )

    qdrant_count = get_qdrant_vector_count(collection_name=collection)
    parity_verified = qdrant_count == len(records)

    if verify_parity and parity_verified:
        qdrant_records = get_qdrant_vectors(collection_name=collection)
        sqlite_ids = {(r["document_id"], r["chunk_index"], r["page_number"]) for r in records}
        qdrant_ids = {(r["document_id"], r["chunk_index"], r["page_number"]) for r in qdrant_records}
        parity_verified = sqlite_ids == qdrant_ids

        # Spot-check top-1 retrieval parity using the first stored embedding.
        sample_vec = records[0]["embedding"]
        sqlite_top = max(
            records,
            key=lambda item: cosine_similarity(sample_vec, item["embedding"]),
        )
        qdrant_hits = search_qdrant_vectors(
            query_embedding=sample_vec,
            top_k=1,
            min_similarity=-1.0,
            collection_name=collection,
        )
        if not qdrant_hits or (
            qdrant_hits[0]["document_id"] != sqlite_top["document_id"]
            or qdrant_hits[0]["chunk_index"] != sqlite_top["chunk_index"]
        ):
            parity_verified = False

    return {
        "status": "migrated",
        "collection": collection,
        "sqlite_count": len(records),
        "migrated_count": migrated,
        "qdrant_count": qdrant_count,
        "parity_verified": parity_verified,
    }


if __name__ == "__main__":
    summary = migrate_sqlite_to_qdrant()
    print(json.dumps(summary, indent=2))
