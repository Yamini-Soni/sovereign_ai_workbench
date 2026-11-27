# Sovereign On-Premise Agentic AI Workbench

A local-first, privacy-focused multimodal AI workbench built with **FastAPI**, **Ollama** (`llama3.2:3b` and `qwen2.5vl:3b`), **FastEmbed** (`sentence-transformers/all-MiniLM-L6-v2`), **SQLite / Qdrant** vector storage, bounded agent orchestration, and an integrated web UI.

## Key Capabilities

- **Local Document Ingestion**: Validates `.pdf`, `.txt`, `.png`, `.jpg`, and `.jpeg` uploads (20 MB limit, SHA-256 deduplication), extracts page-aware PDF text via PyMuPDF, chunks text with overlap, and persists 384-dimensional embeddings.
- **Hybrid RAG & Claim Verification**: Supports semantic cosine search (`app/retrieval/semantic_search.py`), BM25 lexical search (`app/retrieval/lexical_search.py`), Reciprocal Rank Fusion (`app/retrieval/hybrid_search.py`), neighboring-chunk expansion, sentence-level citations, deterministic policy/password verification, and `verify_answer_claims` support checks.
- **Bounded Agent Orchestration**: Iterative local-LLM planner (`app/agents/orchestrator.py`) constrained by an explicit tool allowlist (`app/agents/tool_registry.py`), strict schema validation (`app/agents/argument_validator.py`), safe AST arithmetic (`_calculate`), and redacted JSONL audit logging (`app/agents/tool_executor.py`).
- **Multimodal Vision Integration**: Local image scene, VQA, and document-layout understanding (`app/vision/vision_service.py`), combined image-and-policy agent workflows (`POST /api/v1/agent/run-with-image`), and on-demand image-document indexing (`POST /api/v1/documents/{document_id}/index-image`).
- **Security & Local-Only Controls**: Path-traversal and filename sanitization, prompt-injection detection/neutralization, loopback-only endpoint validation (`app/security/network_guard.py`), and log redaction (`SensitiveDataFilter`).
- **Web UI & Optional Streamlit Client**: Built-in Web UI served directly by FastAPI at `/ui`, plus an optional Streamlit frontend (`frontend/streamlit_app.py`).
- **Qdrant Migration Support**: Default SQLite vector store (`app/retrieval/vector_store.py`) with configurable Qdrant REST backend (`app/retrieval/qdrant_store.py`) and parity-checked migration (`scripts/migrate_to_qdrant.py`).

## Quick Start

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) running locally on `http://localhost:11434`

Pull the default local models:
```bash
ollama pull llama3.2:3b
ollama pull qwen2.5vl:3b
```

### 2. Environment Setup
```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### 3. Run the Application
```bash
uvicorn app.main:app --reload
```
- **Web UI**: `http://127.0.0.1:8000/ui`
- **Interactive API Docs**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/health`

Optional Streamlit UI (requires `streamlit` and `requests` installed):
```bash
streamlit run frontend/streamlit_app.py
```

## API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/` | Root metadata & UI link |
| `GET` | `/ui` | Built-in Web UI |
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/system/status` | Active models, vector backend, and component status |
| `GET` | `/api/v1/system/security` | Security posture and audit log summary |
| `GET` | `/api/v1/system/models` | Discover installed local Ollama models |
| `PUT` | `/api/v1/system/models` | Switch active text and vision models |
| `POST` | `/api/v1/documents/upload` | Upload, validate, extract, chunk, and embed a document |
| `GET` | `/api/v1/documents/` | List uploaded documents |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete a document and its local vectors/artifacts |
| `POST` | `/api/v1/documents/{document_id}/index-image` | Extract & index an `awaiting_ocr` image document via VLM |
| `POST` | `/api/v1/qa/ask` | Grounded RAG QA (`semantic` or `hybrid` mode) |
| `POST` | `/api/v1/agent/run` | Execute bounded text-only agent |
| `POST` | `/api/v1/agent/run-with-image` | Execute bounded multimodal agent with uploaded image |
| `POST` | `/api/v1/vision/analyze` | Analyze image (`scene`, `vqa`, or `document` task) |

## Testing & Benchmarking

Run the full automated test suite:
```bash
python -m pytest -v
```

Run the expanded benchmark harness (offline deterministic dry-run or live Ollama mode):
```bash
python -m scripts.prepare_benchmark_corpus
python -m scripts.run_full_benchmark --dry-run
python -m scripts.run_full_benchmark --retrieval-mode hybrid
```

Run the live local-model security validation (it checks injection detection,
local-only endpoint blocking, tool safety, and audit-log privacy):
```bash
python -m scripts.validate_security_live
```

Migrate vectors from SQLite to a running local Qdrant instance (`http://127.0.0.1:6333`):
```bash
python -m scripts.migrate_to_qdrant
```
