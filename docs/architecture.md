# System Architecture

## 1. Layered Design

The workbench enforces clear separation of concerns across eight logical layers:

1. **Presentation Layer (`app/static/`, `frontend/streamlit_app.py`)**
   - Built-in web UI at `/ui` and optional Streamlit client.
   - Communicates exclusively via FastAPI REST endpoints; holds no model weights, database connections, or retrieval logic.
2. **API & Schema Layer (`app/main.py`, `app/api/v1/`, `app/models/schemas.py`)**
   - FastAPI routers for `system`, `documents`, `qa`, `agent`, and `vision`.
   - Pydantic request/response validation and threadpool offloading for synchronous Ollama calls.
3. **Security & Privacy Layer (`app/security/`)**
   - `sanitizer.py`: Filename sanitization, UUID/ID validation, safe path containment (`resolve_safe_path`), prompt-injection detection/neutralization, and credential redaction (`SensitiveDataFilter`).
   - `network_guard.py`: Enforces loopback-only endpoints (`localhost`, `127.0.0.1`, `::1`) for Ollama and Qdrant.
   - `audit.py`: Structured JSONL security audit trail without storing sensitive content.
4. **Ingestion Layer (`app/ingestion/`)**
   - `file_validator.py`: Extension allowlist (`.pdf`, `.txt`, `.png`, `.jpg`, `.jpeg`), empty-file rejection, and 20 MB size enforcement.
   - `document_extractor.py`: Page-preserving extraction via PyMuPDF (`fitz`) and UTF-8 text reading.
   - `chunker.py`: Overlapping word-boundary chunking (`chunk_size=250`, `overlap=40` in upload pipeline) preserving `page_number` and `chunk_index`.
5. **Embedding, Retrieval & Verification Layer (`app/retrieval/`)**
   - `embedding_service.py`: Local `sentence-transformers/all-MiniLM-L6-v2` (384-d) via FastEmbed.
   - `semantic_search.py` & `lexical_search.py`: Cosine similarity and BM25 scoring.
   - `hybrid_search.py`: Reciprocal Rank Fusion (`rrf_k=60`) combining dense and lexical ranks.
   - `answer_service.py` & `evidence_verifier.py`: Context expansion, deterministic policy/password fact extraction, local LLM prompting, lexical+numeric citation matching, and `verify_answer_claims`.
6. **Storage Layer (`app/storage/`, `app/retrieval/vector_store.py`, `app/retrieval/qdrant_store.py`)**
   - Atomic JSON stores (`documents_metadata.json`, `model_settings.json`) with thread locks.
   - SQLite vector database (`data/vector_store.db`) as default, with drop-in Qdrant REST backend (`VECTOR_STORE_BACKEND=qdrant`) and parity migration (`scripts/migrate_to_qdrant.py`).
7. **Bounded Agent Layer (`app/agents/`)**
   - `orchestrator.py`: `MAX_STEPS=3` iterative planning loop with calculation-grounding checks and RAG fallback.
   - `tool_registry.py`: Allowlist of 5 read-only/safe tools (`answer_document_question`, `search_documents`, `list_documents`, `calculate`, `analyze_image`).
   - `argument_validator.py` & `tool_executor.py`: Strict schema validation, trusted `image_bytes` injection, and redacted audit logging (`logs/agent_audit.jsonl`).
8. **Local Multimodal AI Layer (`app/vision/vision_service.py`)**
   - Ollama chat integration for `llama3.2:3b` (text/planning) and `qwen2.5vl:3b` (scene, VQA, and document-layout analysis).

## 2. End-to-End Request Flow

```text
Client (/ui or REST)
   │
   ▼
FastAPI Router (/api/v1/*) ──► Input & File Validation + Security Guards
   │
   ├──► /documents/upload ──► Extract ──► Chunk ──► FastEmbed ──► SQLite/Qdrant + Metadata JSON
   ├──► /qa/ask           ──► Semantic/Hybrid Search ──► Context Sanitizer ──► Verifier / Ollama ──► Cited Answer
   ├──► /vision/analyze   ──► Magic-byte check ──► Ollama Vision (qwen2.5vl:3b)
   └──► /agent/run*       ──► Planner (Ollama) ──► Allowlist + Validator ──► Executor ──► Audit Log
```
