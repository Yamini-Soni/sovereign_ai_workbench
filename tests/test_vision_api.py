
import urllib.error

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1 import vision


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_image():
    # The endpoint checks the JPEG signature.
    # Inference is mocked, so a complete JPEG is unnecessary.
    return b"\xff\xd8\xff" + b"test-image-data"


def mock_analysis(image_bytes, task, question):
    return {
        "model": "qwen2.5vl:3b",
        "task": task,
        "description": "Mock image analysis result.",
    }


def test_valid_scene_image_returns_success(
    client,
    monkeypatch,
    valid_image,
):
    monkeypatch.setattr(
        vision,
        "analyze_image",
        mock_analysis,
    )

    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "test.jpg",
                valid_image,
                "image/jpeg",
            )
        },
        data={"task": "scene"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["filename"] == "test.jpg"
    assert data["model"] == "qwen2.5vl:3b"
    assert data["task"] == "scene"
    assert data["description"] == "Mock image analysis result."


def test_valid_vqa_image_forwards_question(
    client,
    monkeypatch,
    valid_image,
):
    received = {}

    def mock_vqa(image_bytes, task, question):
        received["image_bytes"] = image_bytes
        received["task"] = task
        received["question"] = question

        return {
            "model": "qwen2.5vl:3b",
            "task": task,
            "description": "Three books are visible.",
        }

    monkeypatch.setattr(
        vision,
        "analyze_image",
        mock_vqa,
    )

    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "books.jpg",
                valid_image,
                "image/jpeg",
            )
        },
        data={
            "task": "vqa",
            "question": "How many books are visible?",
        },
    )

    assert response.status_code == 200
    assert response.json()["description"] == (
        "Three books are visible."
    )

    assert received["image_bytes"] == valid_image
    assert received["task"] == "vqa"
    assert received["question"] == "How many books are visible?"


def test_vqa_without_question_returns_400(
    client,
    valid_image,
):
    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "books.jpg",
                valid_image,
                "image/jpeg",
            )
        },
        data={"task": "vqa"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "A question is required for the vqa task."
    )


def test_unsupported_task_returns_400(
    client,
    valid_image,
):
    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "books.jpg",
                valid_image,
                "image/jpeg",
            )
        },
        data={"task": "unknown"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported vision task."
    )


def test_invalid_image_signature_returns_400(client):
    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "fake.jpg",
                b"this is not a real image",
                "image/jpeg",
            )
        },
        data={"task": "scene"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The uploaded file is not a valid JPEG or PNG image."
    )


def test_unsupported_extension_returns_400(client):
    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "document.pdf",
                b"%PDF-1.4 test",
                "application/pdf",
            )
        },
        data={"task": "scene"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only JPG, JPEG and PNG images are supported."
)


def test_oversized_image_returns_400(
    client,
    monkeypatch,
):
    monkeypatch.setattr(vision, "MAX_FILE_SIZE", 5)

    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "large.jpg",
                b"\xff\xd8\xff123456",
                "image/jpeg",
            )
        },
        data={"task": "scene"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Image exceeds the 20 MB limit."
    )


def test_ollama_unavailable_returns_503(
    client,
    monkeypatch,
    valid_image,
):
    def mock_failure(image_bytes, task, question):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(
        vision,
        "analyze_image",
        mock_failure,
    )

    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "test.jpg",
                valid_image,
                "image/jpeg",
            )
        },
        data={"task": "scene"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Local vision model is unavailable."
    )


def test_vision_timeout_returns_504(
    client,
    monkeypatch,
    valid_image,
):
    def mock_timeout(image_bytes, task, question):
        raise TimeoutError("Inference timed out")

    monkeypatch.setattr(
        vision,
        "analyze_image",
        mock_timeout,
    )

    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "test.jpg",
                valid_image,
                "image/jpeg",
            )
        },
        data={"task": "scene"},
    )

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "Vision inference timed out."
    )


def test_vision_model_failure_returns_502(
    client,
    monkeypatch,
    valid_image,
):
    def mock_failure(image_bytes, task, question):
        raise RuntimeError("Model inference failed")

    monkeypatch.setattr(
        vision,
        "analyze_image",
        mock_failure,
    )

    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "file": (
                "test.jpg",
                valid_image,
                "image/jpeg",
            )
        },
        data={"task": "scene"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Vision model inference failed."
    )
