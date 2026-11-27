
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

    # Local service endpoints and storage backend
    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )
    VECTOR_STORE_BACKEND = os.getenv(
        "VECTOR_STORE_BACKEND",
        "sqlite",
    ).strip().lower()
    QDRANT_URL = os.getenv(
        "QDRANT_URL",
        "http://127.0.0.1:6333",
    )
    QDRANT_COLLECTION = os.getenv(
        "QDRANT_COLLECTION",
        "sovereign_documents",
    )
    OFFLINE_MODE = (
        os.getenv("OFFLINE_MODE", "false").lower() == "true"
    )


settings = Settings()
