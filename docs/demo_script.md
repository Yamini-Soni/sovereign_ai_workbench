# Live Demonstration Script

## Pre-Demo Checklist
1. Verify Ollama is running locally: `ollama list` (confirm `llama3.2:3b` and `qwen2.5vl:3b` are present).
2. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```
3. Open the Web UI at `http://127.0.0.1:8000/ui` (and keep `http://127.0.0.1:8000/docs` in a second tab).

## Step-by-Step Demo Walkthrough

### 1. System & Security Posture Check
- Open the **Models & Security** tab in `/ui`.
- Point out `Local-Only Inference: Verified Local`, `Prompt-Injection Guard: Active`, `Log Redaction Filter: Active`, and the installed Ollama models.

### 2. Document Upload & Indexing
- Open the **Documents** tab.
- Upload `benchmarks/synthetic_industrial_corpus/nexatech_company_policy.txt` and `industrial_maintenance_manual.txt`.
- Show SHA-256 deduplication by re-uploading the same file (`409` duplicate warning).

### 3. Grounded Document QA & Citations
- Open the **Document QA** tab.
- Ask: `"What is the minimum password length?"` -> Demonstrate deterministic verification (`12 characters`) + `[filename | page X]` citation.
- Switch Retrieval Mode to **Hybrid Search** and ask: `"What is the normal operating pressure range for centrifugal pump P-204?"` -> Expand **Retrieved Evidence Sources** and show `40 PSI and 65 PSI` plus the Claim Support badge.
- Ask an unanswerable question: `"What is the CEO's annual salary?"` -> Show clean refusal without hallucination.

### 4. Bounded Agent + Grounded Calculation
- Open the **Agent & Multimodal** tab.
- Submit: `"How many days can employees work remotely per week, and calculate the total over four weeks?"`
- Expand **Execution Trace** to show `answer_document_question` followed by `calculate` (`2 * 4 = 8`) within `MAX_STEPS=3`.

### 5. Multimodal Vision + Policy Query
- Attach an equipment or workspace image in **Agent & Multimodal** and ask: `"Describe the objects in this image. Also, according to the company policy, how should an employee request a laptop replacement?"`
- Show that `qwen2.5vl:3b` analyzes the image while RAG supplies the policy evidence.

### 6. Security Rejection & Document Deletion
- Demonstrate confirmation-protected deletion in **Documents** (filename displayed in modal, UUID sent to `DELETE /api/v1/documents/{document_id}`, vectors removed).
- Show `python -m pytest -v` passing all unit, API, Qdrant migration, benchmark, and security tests.
