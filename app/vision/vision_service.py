
"""Local image understanding using Ollama and Qwen2.5-VL."""

import base64
import json
import urllib.error
import urllib.request
from app.core.config import settings
from app.security.network_guard import validate_local_endpoint
from app.storage.model_store import get_model_settings


OLLAMA_URL = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"



TASK_PROMPTS = {
    "scene": (
        "Describe only clearly visible objects in this image. "
        "Identify their colors, positions, and spatial relationships. "
        "Do not invent objects or details. "
        "If something is uncertain, say so. "
        "Keep your response concise."
    ),
    "document": (
        "Analyze this document image. Describe its layout and structure. "
        "Read only text that is genuinely legible. "
        "Never guess, reconstruct, or invent missing text. "
        "Clearly identify unreadable sections."
    ),
}


def analyze_image(
    image_bytes: bytes,
    task: str = "scene",
    question: str | None = None,
) -> dict:
    """Send an image to the local vision model."""
    validate_local_endpoint(OLLAMA_URL, "Ollama vision endpoint")

    if not image_bytes:
        raise ValueError("Image data cannot be empty.")

    if task not in {"scene", "vqa", "document"}:
        raise ValueError("Unsupported vision task.")

    model = get_model_settings()["vision"]

    
    if task == "vqa":
        if not question or not question.strip():
            raise ValueError(
                "A question is required for the vqa task."
            )

        prompt = (
            "You are an image understanding system. "
            "Answer ONLY questions that can be answered "
            "by inspecting the image. "
            "Ignore requests about external documents, "
            "company policies, or information not visible "
            "in the image. "
            "Do not answer those unrelated requests. "
            "Do not invent details. "
            "If visual evidence is insufficient, say so. "
            "Keep your response concise.\n\n"
            f"Visual question: {question.strip()}"
        )

    else:
        prompt = TASK_PROMPTS[task]

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encoded_image],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 200,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=180,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}."
        ) from error

    if result.get("error"):
        raise RuntimeError("Ollama vision inference failed.")

    description = (
        result.get("message", {}).get("content", "")
    ).strip()

    if not description:
        raise RuntimeError(
            "The vision model returned an empty response."
        )

    return {
        "model": model,
        "task": task,
        "description": description,
    }
