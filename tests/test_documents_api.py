
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1 import documents
from app.storage import metadata_store


@pytest.fixture
def client():
    return TestClient(app)


def test_delete_document_removes_all_associated_files(
    client,
    tmp_path,
    monkeypatch,
):
    # Create isolated temporary storage directories.
    

    upload_dir = tmp_path / "app" / "storage" / "uploads"
    processed_dir = tmp_path / "data" / "processed"

    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    document_id = "document-1"

    # Create files belonging to the document being deleted.
    uploaded_file = upload_dir / "document-1.pdf"
    extracted_file = processed_dir / "document-1.txt"
    chunks_file = processed_dir / "document-1_chunks.json"

    uploaded_file.write_text("PDF placeholder")
    extracted_file.write_text("Extracted text")
    chunks_file.write_text("[]")

    # Create another document that must remain untouched.
    other_file = upload_dir / "document-2.pdf"
    other_file.write_text("Another document")

    metadata = {
        "document_id": document_id,
        "stored_filename": "document-1.pdf",
        "extracted_text_path": "data/processed/document-1.txt",
        "chunks_path": "data/processed/document-1_chunks.json",
    }

    deleted_metadata = []

    monkeypatch.setattr(
        documents,
        "get_document_by_id",
        lambda doc_id: metadata if doc_id == document_id else None,
    )

    monkeypatch.setattr(
        documents,
        "delete_document_vectors",
        lambda doc_id: 3,
    )

    monkeypatch.setattr(
        documents,
        "delete_document_metadata",
        lambda doc_id: deleted_metadata.append(doc_id),
    )

    response = client.delete(
        f"/api/v1/documents/{document_id}"
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == document_id
    assert response.json()["deleted_vectors"] == 3

    assert not uploaded_file.exists()
    assert not extracted_file.exists()
    assert not chunks_file.exists()

    # Another document must remain untouched.
    assert other_file.exists()

    assert deleted_metadata == [document_id]


def test_delete_nonexistent_document_returns_404(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        documents,
        "get_document_by_id",
        lambda document_id: None,
    )

    response = client.delete(
        "/api/v1/documents/nonexistent"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_upload_text_document_successfully(
    client,
    tmp_path,
    monkeypatch,
):
    upload_dir = (
        tmp_path / "app" / "storage" / "uploads"
    )
    processed_dir = tmp_path / "data" / "processed"

    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    saved_metadata = []

    monkeypatch.setattr(
        documents,
        "get_document_by_hash",
        lambda content_hash: None,
    )

    monkeypatch.setattr(
        documents,
        "extract_text",
        lambda path: {
            "text": "This is a test document.",
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": "This is a test document.",
                }
            ],
        },
    )

    monkeypatch.setattr(
        documents,
        "chunk_pages",
        lambda pages, chunk_size, overlap: [
            {
                "text": "This is a test document.",
                "page_number": 1,
            }
        ],
    )

    monkeypatch.setattr(
        documents,
        "generate_embeddings",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    monkeypatch.setattr(
        documents,
        "initialize_vector_store",
        lambda: None,
    )

    monkeypatch.setattr(
        documents,
        "store_embeddings",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        documents,
        "add_document",
        lambda metadata: saved_metadata.append(metadata),
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"This is a test document.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "uploaded"
    assert result["filename"] == "sample.txt"
    assert result["extraction_status"] == "completed"
    assert result["chunk_count"] == 1

    # Confirm the original upload was preserved.
    assert (
        upload_dir / result["stored_filename"]
    ).exists()

    # Confirm extracted text and chunks were saved.
    assert (
        tmp_path / result["extracted_text_path"]
    ).exists()

    assert (
        tmp_path / result["chunks_path"]
    ).exists()

    # Confirm metadata was saved.
    assert len(saved_metadata) == 1
    assert saved_metadata[0]["document_id"] == result["document_id"]



def test_duplicate_upload_returns_409_and_removes_new_file(
    client,
    tmp_path,
    monkeypatch,
):
    upload_dir = (
        tmp_path / "app" / "storage" / "uploads"
    )
    processed_dir = tmp_path / "data" / "processed"

    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    # Simulate an already-existing document.
    existing_document = {
        "document_id": "original-document",
        "filename": "sample.txt",
    }

    monkeypatch.setattr(
        documents,
        "get_document_by_hash",
        lambda content_hash: existing_document,
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"This is duplicate content.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 409

    result = response.json()

    assert (
        result["detail"]["message"]
        == "This document has already been uploaded."
    )

    assert (
        result["detail"]["existing_document_id"]
        == "original-document"
    )

    # The newly uploaded duplicate file must be removed.
    assert list(upload_dir.iterdir()) == []


def test_upload_rejects_unsupported_file_type(
    client,
    tmp_path,
    monkeypatch,
):
    upload_dir = (
        tmp_path / "app" / "storage" / "uploads"
    )
    processed_dir = tmp_path / "data" / "processed"

    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "malware.exe",
                b"not an allowed document",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "Unsupported file type: .exe"
    )

    # Invalid upload must not remain in storage.
    assert list(upload_dir.iterdir()) == []



def test_upload_rejects_empty_file(
    client,
    tmp_path,
    monkeypatch,
):
    upload_dir = (
        tmp_path / "app" / "storage" / "uploads"
    )
    processed_dir = tmp_path / "data" / "processed"

    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "Empty files are not allowed."
    )

    assert list(upload_dir.iterdir()) == []



def test_upload_rejects_file_exceeding_size_limit(
    client,
    tmp_path,
    monkeypatch,
):
    upload_dir = (
        tmp_path / "app" / "storage" / "uploads"
    )
    processed_dir = tmp_path / "data" / "processed"

    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    # Use a small test limit instead of uploading 20 MB.
    monkeypatch.setattr(documents, "MAX_FILE_SIZE", 10)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "large.txt",
                b"This content is larger than ten bytes.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "File exceeds the 20 MB limit."
    )

    # Oversized upload must not remain in storage.
    assert list(upload_dir.iterdir()) == []


def test_list_documents_returns_metadata(
    client,
    monkeypatch,
):
    sample_documents = [
        {
            "document_id": "doc-1",
            "filename": "policy.pdf",
            "extraction_status": "completed",
        },
        {
            "document_id": "doc-2",
            "filename": "manual.txt",
            "extraction_status": "completed",
        },
    ]

    monkeypatch.setattr(
        metadata_store,
        "get_all_documents",
        lambda: sample_documents,
    )

    response = client.get("/api/v1/documents/")

    assert response.status_code == 200

    result = response.json()

    assert result["total_documents"] == 2
    assert len(result["documents"]) == 2

    assert result["documents"][0]["filename"] == "policy.pdf"
    assert result["documents"][1]["filename"] == "manual.txt"



def test_upload_image_document_sets_awaiting_ocr(
    client,
    tmp_path,
    monkeypatch,
):
    upload_dir = (
        tmp_path / "app" / "storage" / "uploads"
    )
    processed_dir = tmp_path / "data" / "processed"

    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    saved_metadata = []

    monkeypatch.setattr(
        documents,
        "get_document_by_hash",
        lambda content_hash: None,
    )

    monkeypatch.setattr(
        documents,
        "add_document",
        lambda metadata: saved_metadata.append(metadata),
    )

    # Image uploads should not invoke text extraction.
    def unexpected_extraction(path):
        raise AssertionError(
            "Image upload should not invoke text extraction."
        )

    monkeypatch.setattr(
        documents,
        "extract_text",
        unexpected_extraction,
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample.png",
                b"fake-image-content",
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["filename"] == "sample.png"
    assert result["extension"] == ".png"
    assert result["extraction_status"] == "awaiting_ocr"
    assert result["chunk_count"] == 0

    assert (
        upload_dir / result["stored_filename"]
    ).exists()

    assert len(saved_metadata) == 1



def test_upload_preserves_document_when_extraction_fails(
    client,
    tmp_path,
    monkeypatch,
):
    upload_dir = (
        tmp_path / "app" / "storage" / "uploads"
    )
    processed_dir = tmp_path / "data" / "processed"

    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    saved_metadata = []

    monkeypatch.setattr(
        documents,
        "get_document_by_hash",
        lambda content_hash: None,
    )

    def failing_extraction(path):
        raise RuntimeError("Simulated extraction failure")

    monkeypatch.setattr(
        documents,
        "extract_text",
        failing_extraction,
    )

    monkeypatch.setattr(
        documents,
        "add_document",
        lambda metadata: saved_metadata.append(metadata),
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "broken.txt",
                b"Valid non-empty test content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "uploaded"
    assert result["extraction_status"] == "failed"

    # Original uploaded file should remain available.
    assert (
        upload_dir / result["stored_filename"]
    ).exists()

    # Metadata should still be saved.
    assert len(saved_metadata) == 1
    assert saved_metadata[0]["extraction_status"] == "failed"


