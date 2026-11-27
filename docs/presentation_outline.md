# Capstone Presentation Outline

1. **Title & Problem Statement**: Why organizations handling confidential manuals and policies cannot blindly send data to cloud LLMs; project objectives for an on-premise multimodal AI workbench.
2. **System Architecture**: 8-layer modular FastAPI architecture separating UI (`/ui`), ingestion, SQLite/Qdrant vector storage, hybrid RAG, bounded agent orchestration, vision service, and security controls.
3. **Document Ingestion & Hybrid RAG Pipeline**: PyMuPDF page-preserving extraction, overlapping chunking, `all-MiniLM-L6-v2` 384-d local embeddings, cosine + BM25 RRF fusion, and sentence-level citations.
4. **Evidence Verification & The Q05 Finding**: How Q05 (*minimum password length*) exposed a generation contradiction despite accurate retrieval, leading to `evidence_verifier.py` and `verify_answer_claims`.
5. **Bounded Agentic AI & Multimodal Integration**: `MAX_STEPS=3` orchestrator, 5-tool allowlist, AST-safe calculator, `qwen2.5vl:3b` VQA, and `/api/v1/agent/run-with-image`.
6. **Privacy & Security Controls**: Loopback-only network guard, path-traversal guards, prompt-injection sanitization, and redacted audit trails (`logs/agent_audit.jsonl`, `logs/security_audit.jsonl`).
7. **Evaluation Results & Hardware Footprint**: Baseline Experiment 01 (14/15 = 93.3% pass rate, 5/5 refusals, 2.67s median latency, 2539 MiB VRAM on RTX 3050) and expanded 26-QA + 4-agent benchmark suite (`scripts/run_full_benchmark.py`).
8. **Live Demo & Conclusion**: Walkthrough of `/ui`, remaining production considerations (multi-user auth, larger quantized models), and Q&A.
