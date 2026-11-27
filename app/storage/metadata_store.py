
import json
from pathlib import Path
from threading import Lock

from app.core.config import BASE_DIR


METADATA_FILE = (
    BASE_DIR / "app" / "storage" / "documents_metadata.json"
)

METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

_store_lock = Lock()


def _load_metadata_unlocked() -> list:
    """Load metadata. Caller must hold the lock."""

    if not METADATA_FILE.exists():
        return []

    try:
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError:
        raise RuntimeError(
            "Document metadata file is corrupted."
        )


def _save_metadata_unlocked(documents: list) -> None:
    """Save metadata atomically. Caller must hold the lock."""

    temporary_file = METADATA_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    temporary_file.replace(METADATA_FILE)


def load_metadata() -> list:
    """Load document metadata from local JSON storage."""

    with _store_lock:
        return _load_metadata_unlocked()


def save_metadata(documents: list) -> None:
    """Save document metadata atomically."""

    with _store_lock:
        _save_metadata_unlocked(documents)


def add_document(metadata: dict) -> None:
    """Add a document record."""

    with _store_lock:
        documents = _load_metadata_unlocked()
        documents.append(metadata)
        _save_metadata_unlocked(documents)


def get_document_by_hash(content_hash: str):
    """Find an existing document using its SHA-256 fingerprint."""

    with _store_lock:
        documents = _load_metadata_unlocked()

        for document in documents:
            if document.get("content_hash") == content_hash:
                return document

        return None


def get_all_documents() -> list:
    """Return all stored document records."""

    return load_metadata()


def get_document_by_id(document_id: str):
    """Find a document using its document ID."""

    with _store_lock:
        documents = _load_metadata_unlocked()

        for document in documents:
            if document.get("document_id") == document_id:
                return document

        return None


def delete_document_metadata(document_id: str) -> bool:
    """Delete one document's metadata record."""

    with _store_lock:
        documents = _load_metadata_unlocked()

        updated_documents = [
            document
            for document in documents
            if document.get("document_id") != document_id
        ]

        if len(updated_documents) == len(documents):
            return False

        _save_metadata_unlocked(updated_documents)

        return True


def update_document_metadata(document_id: str, updates: dict) -> dict | None:
    """Update fields on an existing document metadata record atomically."""

    with _store_lock:
        documents = _load_metadata_unlocked()

        for index, document in enumerate(documents):
            if document.get("document_id") == document_id:
                updated = {**document, **updates}
                documents[index] = updated
                _save_metadata_unlocked(documents)
                return updated

        return None

