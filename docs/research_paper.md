# Sovereign On-Premise Agentic AI Workbench Using Open-Weight Multimodal LLMs

## Abstract
Organizations handling confidential operational manuals, HR policies, and industrial inspection records face strict barriers when adopting cloud-hosted Large Language Models (LLMs), including data egress risks, recurring API costs, and external network dependencies. This capstone project presents the **Sovereign On-Premise Agentic AI Workbench**, a modular, local-first prototype combining document ingestion, hybrid Retrieval-Augmented Generation (RAG), deterministic evidence verification, bounded agent tool orchestration, and local vision-language analysis. Built with FastAPI, Ollama (`llama3.2:3b` and `qwen2.5vl:3b`), `all-MiniLM-L6-v2` embeddings, and SQLite/Qdrant vector storage, the system operates entirely on local hardware (evaluated on a laptop with 16 GB RAM and an NVIDIA RTX 3050 6 GB VRAM GPU). Preliminary baseline evaluation across 15 policy questions achieved a 93.3% keyword-based pass rate (9/10 answerable, 5/5 unanswerable refusals) with a 2.67s median response time while surfacing a key generation-level contradiction (Q05) that motivated separate retrieval and claim-verification layers.

## 1. Introduction
Enterprise knowledge repositories frequently reside in unstructured PDFs, technical SOPs, and equipment photographs. Keyword search lacks semantic recall, whereas external cloud AI services introduce privacy and compliance risks. This project investigates how compact, open-weight text and vision models can be integrated into a secure, verifiable local workbench without requiring cloud inference endpoints.

## 2. System Architecture & Methodology
The workbench is structured into eight decoupled layers:
- **Ingestion & Chunking**: File validation (`.pdf`, `.txt`, `.png`, `.jpg`, `.jpeg`, 20 MB cap, SHA-256 deduplication), page-preserving PyMuPDF extraction, and overlapping chunking (`chunk_size=250`, `overlap=40` in the upload pipeline).
- **Embeddings & Storage**: Local 384-dimensional embeddings (`sentence-transformers/all-MiniLM-L6-v2` via FastEmbed) persisted in SQLite (`data/vector_store.db`) with a parity-verified Qdrant REST backend (`app/retrieval/qdrant_store.py`).
- **Hybrid Retrieval & Verification**: Cosine semantic retrieval paired with BM25 lexical search via Reciprocal Rank Fusion (`rrf_k=60`). `answer_service.py` combines neighboring-chunk expansion, deterministic verification (`verify_policy_facts`, `verify_password_length`), sentence-level evidence citation matching (`add_evidence_citations`), and `verify_answer_claims` numeric/lexical support verification.
- **Bounded Agentic Layer**: Rather than granting unrestricted shell or code execution, `orchestrator.py` limits execution to `MAX_STEPS=3` over five allowlisted tools (`answer_document_question`, `search_documents`, `list_documents`, AST-restricted `calculate`, and `analyze_image`).
- **Multimodal Vision Layer**: `vision_service.py` invokes `qwen2.5vl:3b` locally for scene description, VQA, document-layout reading, and combined image-and-policy agent queries (`/api/v1/agent/run-with-image`).
- **Security Controls**: Loopback-only endpoint enforcement (`validate_local_endpoint`), path-traversal rejection, prompt-injection sanitization, and redacted JSONL audit logging.

## 3. Experimental Setup & Preliminary Results
### 3.1 Baseline Experiment 01 (Recorded)
Evaluated on the NexaTech Company Policy dataset (15 questions: 10 answerable, 5 unanswerable) using `llama3.2:3b`, `all-MiniLM-L6-v2`, SQLite, Top-K=3, and minimum similarity 0.30 (`docs/baseline_experiment.md`):
- **Answerable accuracy**: 9/10 (90.0%)
- **Unanswerable refusal accuracy**: 5/5 (100.0%)
- **Overall keyword-based pass rate**: 14/15 (93.3%)
- **Median end-to-end latency**: 2.67 seconds
- **Sampled RTX 3050 GPU telemetry**: 2539 MiB / 6144 MiB VRAM used, 99% peak utilization, 51°C max temperature, 36.19 W peak power.

### 3.2 Key Finding: Retrieval vs. Generation Decoupling
Question Q05 (*"What is the minimum password length?"*) retrieved the correct policy chunk containing `"12 characters"`, yet the 3B model generated a contradictory refusal. This demonstrated that retrieval accuracy and generation faithfulness are distinct failure modes requiring independent verification (`verify_password_length` and `verify_answer_claims`).

### 3.3 Expanded Benchmark Suite
To extend evaluation beyond a single 15-question document, `benchmarks/benchmark_cases.json` introduces 26 QA cases across a multi-document synthetic industrial corpus (HR/IT policy, centrifugal pump P-204 maintenance manual, and an archived adversarial draft testing retrieval contamination and prompt injection) plus 4 agent tool-selection/safety cases, measured by `scripts/run_full_benchmark.py` (Recall@K, Precision@K, MRR@K, refusal accuracy, claim support ratio, median/p95 latency, and RAM/VRAM usage).

## 4. Limitations & Future Work
- **Compact Model Reasoning**: 3B-parameter models fit 6 GB VRAM laptops but exhibit occasional JSON planner errors and numeric contradictions compared to larger models.
- **Image OCR Indexing**: Image documents uploaded via `/api/v1/documents/upload` require explicit triggering of `/api/v1/documents/{id}/index-image` (or UI button) to convert VLM output into searchable chunks.
- **Multi-User Auth**: Future production hardening should add user authentication, RBAC per document collection, and OS-level container network isolation.

## 5. Conclusion
The Sovereign On-Premise Agentic AI Workbench demonstrates a practical, reproducible foundation for local document intelligence, bounded multimodal agent workflows, and auditable privacy safeguards on consumer-grade hardware.
