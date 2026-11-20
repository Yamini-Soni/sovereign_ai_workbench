
"""Thread-safe local storage for selected AI models."""

import json
from pathlib import Path
from threading import Lock

from app.core.config import BASE_DIR, settings


MODEL_SETTINGS_FILE = (
    BASE_DIR / "app" / "storage" / "model_settings.json"
)

MODEL_SETTINGS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

_store_lock = Lock()

DEFAULT_MODELS = {
    "text": settings.TEXT_MODEL,
    "vision": settings.VISION_MODEL,
}


def _load_unlocked() -> dict:
    """Load model settings. Caller must hold the lock."""

    if not MODEL_SETTINGS_FILE.exists():
        return DEFAULT_MODELS.copy()

    try:
        with MODEL_SETTINGS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            stored = json.load(file)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Model settings file is corrupted."
        ) from error

    if not isinstance(stored, dict):
        raise RuntimeError(
            "Model settings must contain a JSON object."
        )

    return {
        **DEFAULT_MODELS,
        **{
            key: stored[key]
            for key in ("text", "vision")
            if isinstance(stored.get(key), str)
            and stored[key].strip()
        },
    }


def get_model_settings() -> dict:
    """Return the currently selected models."""

    with _store_lock:
        return _load_unlocked()


def save_model_settings(
    text_model: str,
    vision_model: str,
) -> dict:
    """Persist model selections using an atomic file replacement."""

    models = {
        "text": text_model,
        "vision": vision_model,
    }

    temporary_file = MODEL_SETTINGS_FILE.with_suffix(".tmp")

    with _store_lock:
        with temporary_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                models,
                file,
                indent=4,
                ensure_ascii=False,
            )

        temporary_file.replace(MODEL_SETTINGS_FILE)

    return models
