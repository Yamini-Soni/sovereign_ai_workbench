"""Tests for Qdrant vector store adapter and SQLite-to-Qdrant migration parity."""

import json
import sqlite3

import pytest

from app.core.config import settings
from app.retrieval import qdrant_store, vector_store
from scripts import migrate_to_qdrant


class FakeQdrantBackend:
    """In-memory fake Qdrant REST backend for deterministic unit testing."""

    def __init__(self):
        self.collections = {}

    def handle(self, method: str, path: str, payload: dict | None = None, timeout: int = 30) -> dict:
        parts = [p for p in path.split("?")[0].split("/") if p]
        # /collections/{name}
        if len(parts) == 2 and parts[0] == "collections":
            name = parts[1]
            if method == "GET":
                if name not in self.collections:
                    import urllib.error
                    raise urllib.error.HTTPError(path, 404, "Not Found", {}, None)
                return {"result": {"status": "green"}}
            if method == "PUT":
                self.collections[name] = {}
                return {"result": True}

        # /collections/{name}/points...
        name = parts[1]
        store = self.collections.setdefault(name, {})

        if parts[2] == "points" and len(parts) == 3 and method == "PUT":
            for pt in (payload or {}).get("points", []):
                store[pt["id"]] = pt
            return {"result": {"status": "completed"}}

        if len(parts) == 4 and parts[3] == "count":
            return {"result": {"count": len(store)}}

        if len(parts) == 4 and parts[3] == "scroll":
            doc_filter = None
            must = ((payload or {}).get("filter") or {}).get("must") or []
            if must:
                doc_filter = must[0]["match"]["value"]
            pts = [
                pt for pt in store.values()
                if doc_filter is None or pt["payload"].get("document_id") == doc_filter
            ]
            return {"result": {"points": pts, "next_page_offset": None}}

        if len(parts) == 4 and parts[3] == "search":
            qvec = (payload or {}).get("vector") or []
            hits = []
            for pt in store.values():
                vec = pt["vector"]
                score = sum(a * b for a, b in zip(qvec, vec))
                hits.append({
                    "id": pt["id"],
                    "score": score,
                    "payload": pt["payload"],
                })
            hits.sort(key=lambda x: x["score"], reverse=True)
            return {"result": hits[: (payload or {}).get("limit", 5)]}

        if len(parts) == 4 and parts[3] == "delete":
            must = ((payload or {}).get("filter") or {}).get("must") or []
            doc_id = must[0]["match"]["value"]
            to_del = [pid for pid, pt in store.items() if pt["payload"].get("document_id") == doc_id]
            for pid in to_del:
                del store[pid]
            return {"result": {"status": "completed"}}

        return {}


def test_qdrant_store_and_migration_parity(tmp_path, monkeypatch):
    fake_qdrant = FakeQdrantBackend()
    monkeypatch.setattr(qdrant_store, "_qdrant_request", fake_qdrant.handle)

    db_path = tmp_path / "vector_store.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE document_vectors (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )

    emb_a = [1.0] + [0.0] * 383
    emb_b = [0.0, 1.0] + [0.0] * 382
    conn.executemany(
        "INSERT INTO document_vectors VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("doc1_0", "doc1", 0, 1, "Annual leave is 20 days.", json.dumps(emb_a)),
            ("doc2_0", "doc2", 0, 2, "Pump pressure is 50 PSI.", json.dumps(emb_b)),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        migrate_to_qdrant,
        "get_all_documents",
        lambda: [
            {"document_id": "doc1", "filename": "policy.pdf"},
            {"document_id": "doc2", "filename": "manual.pdf"},
        ],
    )

    summary = migrate_to_qdrant.migrate_sqlite_to_qdrant(
        db_path=db_path,
        collection_name="test_collection",
        verify_parity=True,
    )

    assert summary["status"] == "migrated"
    assert summary["sqlite_count"] == 2
    assert summary["qdrant_count"] == 2
    assert summary["parity_verified"] is True

    # Verify vector_store dispatch when VECTOR_STORE_BACKEND == "qdrant"
    monkeypatch.setattr(settings, "VECTOR_STORE_BACKEND", "qdrant")
    monkeypatch.setattr(settings, "QDRANT_COLLECTION", "test_collection")
    assert vector_store.get_vector_count() == 2
    deleted = vector_store.delete_document_vectors("doc1")
    assert deleted == 1
    assert vector_store.get_vector_count() == 1
