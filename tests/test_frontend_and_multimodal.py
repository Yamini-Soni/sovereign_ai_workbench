"""Tests for Web UI route, hybrid QA mode, image document indexing, and claim verification."""

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import documents, qa
from app.main import app
from app.retrieval.evidence_verifier import verify_answer_claims


@pytest.fixture
def client():
    return TestClient(app)


def test_root_and_ui_routes(client):
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert root_resp.json()["ui_url"] == "/ui"

    ui_resp = client.get("/ui")
    assert ui_resp.status_code == 200
    assert "Sovereign AI Workbench" in ui_resp.text


def test_qa_hybrid_retrieval_mode_forwarded(client, monkeypatch):
    captured = {}

    def mock_answer(question, document_id=None, retrieval_mode="semantic"):
        captured["retrieval_mode"] = retrieval_mode
        return {
            "answer": "Pump pressure must stay between 40 PSI and 65 PSI.",
            "answer_method": "local_llm",
            "verified_evidence": None,
            "claim_verification": {"supported": True, "support_ratio": 1.0},
            "sources": [],
        }

    monkeypatch.setattr(qa, "answer_question", mock_answer)

    response = client.post(
        "/api/v1/qa/ask",
        json={
            "question": "What is the pump pressure?",
            "retrieval_mode": "hybrid",
        },
    )
    assert response.status_code == 200
    assert captured["retrieval_mode"] == "hybrid"
    assert response.json()["claim_verification"]["supported"] is True


def test_index_uploaded_image_document(client, tmp_path, monkeypatch):
    upload_dir = tmp_path / "app" / "storage" / "uploads"
    processed_dir = tmp_path / "data" / "processed"
    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    monkeypatch.setattr(documents, "BASE_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    img_file = upload_dir / "img-doc-1.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nfake-png")

    doc_record = {
        "document_id": "img-doc-1",
        "filename": "label.png",
        "stored_filename": "img-doc-1.png",
        "extension": ".png",
        "extraction_status": "awaiting_ocr",
        "chunk_count": 0,
    }

    monkeypatch.setattr(documents, "get_document_by_id", lambda doc_id: doc_record if doc_id == "img-doc-1" else None)
    monkeypatch.setattr(
        documents,
        "analyze_image",
        lambda image_bytes, task="document": {
            "model": "qwen2.5vl:3b",
            "task": task,
            "description": "Pump P-204 pressure warning label showing 65 PSI max.",
        },
    )
    monkeypatch.setattr(documents, "generate_embeddings", lambda texts: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(documents, "initialize_vector_store", lambda: None)
    monkeypatch.setattr(documents, "store_embeddings", lambda **kwargs: None)
    monkeypatch.setattr(
        documents,
        "update_document_metadata",
        lambda doc_id, updates: {**doc_record, **updates},
    )

    response = client.post("/api/v1/documents/img-doc-1/index-image")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "indexed"
    assert payload["document"]["extraction_status"] == "completed"
    assert payload["document"]["chunk_count"] >= 1


def test_verify_answer_claims_catches_numeric_contradiction():
    chunks = [
        {"text": "Employees may work remotely up to 2 days per week."}
    ]

    supported = verify_answer_claims(
        "Employees may work remotely up to 2 days per week.",
        chunks,
    )
    assert supported["supported"] is True

    contradicted = verify_answer_claims(
        "Employees may work remotely up to 3 days per week.",
        chunks,
    )
    assert contradicted["supported"] is False
    assert len(contradicted["unsupported_sentences"]) == 1
