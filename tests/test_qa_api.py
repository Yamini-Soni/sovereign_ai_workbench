
import urllib.error

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1 import qa


@pytest.fixture
def client():
    return TestClient(app)


def test_valid_question_returns_answer(client, monkeypatch):
    def mock_answer_question(question, document_id=None):
        return {
            "answer": "Employees receive 20 days of annual leave.",
            "answer_method": "deterministic_policy_evidence_extraction",
            "verified_evidence": [],
            "sources": [],
        }

    monkeypatch.setattr(
        qa,
        "answer_question",
        mock_answer_question,
    )

    response = client.post(
        "/api/v1/qa/ask",
        json={
            "question": "How many days of annual leave?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "20 days" in data["answer"]
    assert data["question"] == "How many days of annual leave?"
    assert data["answer_method"] == (
        "deterministic_policy_evidence_extraction"
    )
    assert "latency_seconds" in data


def test_question_whitespace_is_rejected(client):
    response = client.post(
        "/api/v1/qa/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Question cannot be empty."
    )


def test_empty_question_is_rejected(client):
    response = client.post(
        "/api/v1/qa/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_optional_document_id_is_forwarded(
    client,
    monkeypatch,
):
    received_arguments = {}

    def mock_answer_question(question, document_id=None):
        received_arguments["question"] = question
        received_arguments["document_id"] = document_id

        return {
            "answer": "Test answer.",
            "answer_method": "test",
            "verified_evidence": None,
            "sources": [],
        }

    monkeypatch.setattr(
        qa,
        "answer_question",
        mock_answer_question,
    )

    response = client.post(
        "/api/v1/qa/ask",
        json={
            "question": "Test question",
            "document_id": "doc_policy",
        },
    )

    assert response.status_code == 200
    assert received_arguments["question"] == "Test question"
    assert received_arguments["document_id"] == "doc_policy"


def test_ollama_unavailable_returns_503(
    client,
    monkeypatch,
):
    def mock_answer_question(question, document_id=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(
        qa,
        "answer_question",
        mock_answer_question,
    )

    response = client.post(
        "/api/v1/qa/ask",
        json={"question": "Test question"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Local LLM service is unavailable."
    )
