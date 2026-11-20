
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1 import agent


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_image():
    return b"\xff\xd8\xff" + b"test-image-data"


def test_text_agent_endpoint(client, monkeypatch):
    def mock_run_agent(question, image_bytes=None):
        return {"answer": "Mock agent answer."}

    monkeypatch.setattr(agent, "run_agent", mock_run_agent)

    response = client.post(
        "/api/v1/agent/run",
        json={"question": "Summarize my documents"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Mock agent answer."


def test_image_agent_endpoint(client, monkeypatch, valid_image):
    received = {}

    def mock_run_agent(question, image_bytes=None):
        received["question"] = question
        received["image_bytes"] = image_bytes
        return {"answer": "A book is visible."}

    monkeypatch.setattr(agent, "run_agent", mock_run_agent)

    response = client.post(
        "/api/v1/agent/run-with-image",
        data={"question": "What is in this image?"},
        files={
            "image": ("test.jpg", valid_image, "image/jpeg")
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "A book is visible."
    assert response.json()["image_attached"] is True
    assert response.json()["image_format"] == "jpeg"
    assert received["question"] == "What is in this image?"
    assert received["image_bytes"] == valid_image


def test_image_agent_rejects_unsupported_extension(client, valid_image):
    response = client.post(
        "/api/v1/agent/run-with-image",
        data={"question": "Describe this."},
        files={
            "image": ("test.gif", valid_image, "image/gif")
        },
    )

    assert response.status_code == 415


def test_image_agent_rejects_invalid_signature(client):
    response = client.post(
        "/api/v1/agent/run-with-image",
        data={"question": "Describe this."},
        files={
            "image": ("test.jpg", b"not-an-image", "image/jpeg")
        },
    )

    assert response.status_code == 400


def test_image_agent_rejects_whitespace_question(client, valid_image):
    response = client.post(
        "/api/v1/agent/run-with-image",
        data={"question": "   "},
        files={
            "image": ("test.jpg", valid_image, "image/jpeg")
        },
    )

    assert response.status_code == 400


def test_text_agent_rejects_whitespace_question(client):
    response = client.post(
        "/api/v1/agent/run",
        json={"question": "   "},
    )

    assert response.status_code == 400
