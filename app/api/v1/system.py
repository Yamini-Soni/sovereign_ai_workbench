
"""System status and local model discovery and selection."""

import json
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.storage.model_store import (
    get_model_settings,
    save_model_settings,
)


router = APIRouter(
    prefix="/system",
    tags=["System"],
)

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


class ModelSelection(BaseModel):
    text_model: str = Field(min_length=1)
    vision_model: str = Field(min_length=1)


def _fetch_local_models() -> list[dict]:
    """Fetch installed models from the local Ollama service."""

    request = urllib.request.Request(
        OLLAMA_TAGS_URL,
        headers={"Content-Type": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=10,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except (urllib.error.URLError, TimeoutError) as error:
        raise HTTPException(
            status_code=503,
            detail="Local Ollama service is unavailable.",
        ) from error

    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HTTPException(
            status_code=502,
            detail="Invalid response from local Ollama service.",
        ) from error

    if not isinstance(result, dict) or not isinstance(
        result.get("models"), list
    ):
        raise HTTPException(
            status_code=502,
            detail="Invalid model list from Ollama.",
        )

    return [
        item
        for item in result["models"]
        if isinstance(item, dict) and item.get("name")
    ]


@router.get("/status")
def system_status():
    selected = get_model_settings()

    return {
        "application": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "model_provider": settings.MODEL_PROVIDER,
        "models": {
            "text": selected["text"],
            "vision": selected["vision"],
            "embedding": settings.EMBEDDING_MODEL,
        },
        "components": {
            "document_ingestion": "available",
            "retrieval": "available",
            "llm": "configured",
            "vision": "configured",
            "agent": "configured",
            "security": "not_verified",
        },
    }


@router.get("/models")
def list_local_models():
    """Return installed models and current selections."""

    installed = _fetch_local_models()
    selected = get_model_settings()

    models = [
        {
            "name": item.get("name"),
            "size": item.get("size"),
            "modified_at": item.get("modified_at"),
        }
        for item in installed
    ]

    return {
        "total": len(models),
        "models": models,
        "selected": selected,
    }


@router.put("/models")
def update_model_selection(selection: ModelSelection):
    """Validate and save the selected text and vision models."""

    installed = _fetch_local_models()
    installed_names = {
        item["name"]
        for item in installed
    }

    requested = {
        "text": selection.text_model.strip(),
        "vision": selection.vision_model.strip(),
    }

    missing = [
        name
        for name in requested.values()
        if name not in installed_names
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more selected models are not installed.",
                "missing_models": missing,
            },
        )

    selected = save_model_settings(
        text_model=requested["text"],
        vision_model=requested["vision"],
    )

    return {
        "message": "Model selections updated successfully.",
        "selected": selected,
    }
