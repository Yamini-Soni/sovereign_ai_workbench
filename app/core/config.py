
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_NAME = os.getenv(
        "PROJECT_NAME",
        "Sovereign AI Workbench",
    )
    ENVIRONMENT = os.getenv(
        "ENVIRONMENT",
        "development",
    )
    DEBUG = os.getenv(
        "DEBUG",
        "true",
    ).lower() == "true"

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))

    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "local")

    # Default local Ollama models
    TEXT_MODEL = os.getenv(
        "TEXT_MODEL",
        "llama3.2:3b",
    )
    VISION_MODEL = os.getenv(
        "VISION_MODEL",
        "qwen2.5vl:3b",
    )

    # Local embedding model (not an Ollama chat model)
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


settings = Settings()
