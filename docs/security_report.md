# Security and Privacy Evaluation Report

## 1. Security Posture Summary

Running models locally reduces third-party data exposure, but local inference alone does not guarantee application security. The workbench implements explicit application-level controls across ingestion, retrieval, agent tool execution, network egress, and logging.

## 2. Threat Model & Implemented Controls

| Risk | Implemented Safeguard | Module(s) | Verification |
| :--- | :--- | :--- | :--- |
| **External data transmission** | `validate_local_endpoint` restricts Ollama and Qdrant URLs to `localhost`, `127.0.0.1`, or `::1` | `app/security/network_guard.py`, `answer_service.py`, `vision_service.py`, `orchestrator.py`, `system.py`, `qdrant_store.py` | `test_network_guard_blocks_external_endpoints` |
| **Unsafe file uploads** | Extension allowlist, size limit (20 MB), empty-file rejection, JPEG/PNG magic-byte checks | `app/ingestion/file_validator.py`, `app/api/v1/agent.py`, `app/api/v1/vision.py` | `tests/test_documents_api.py`, `tests/test_vision_api.py`, `tests/test_agent_api.py` |
| **Path traversal** | `sanitize_filename` strips directory prefixes & rejects null bytes; `validate_document_id` enforces safe IDs; `resolve_safe_path` enforces containment | `app/security/sanitizer.py`, `app/api/v1/documents.py`, `app/api/v1/qa.py` | `test_sanitize_filename_strips_traversal_and_rejects_null_bytes`, `test_validate_document_id_rejects_traversal`, `test_resolve_safe_path_blocks_escape` |
| **Prompt injection in documents/prompts** | System prompts treat context as untrusted data; `sanitize_retrieved_context` strips `<system>` tags and logs injection indicators; planner still enforces allowlist | `app/security/sanitizer.py`, `app/retrieval/answer_service.py`, `app/agents/orchestrator.py` | `test_prompt_injection_detection_and_neutralization`, `test_prompt_injected_agent_tool_call_is_blocked` |
| **Unauthorized tool execution** | Allowlist of 5 tools; `argument_validator.py` rejects unexpected types/fields; `analyze_image` only receives server-validated `image_bytes`; AST-restricted `_calculate` | `app/agents/tool_registry.py`, `argument_validator.py`, `tool_executor.py` | `test_tool_executor_blocks_unknown_tool_and_logs_without_values`, `test_calculator_blocks_code_execution_attempts`, `tests/test_orchestrator.py` |
| **Sensitive data in logs** | `SensitiveDataFilter` redacts tokens/keys/passwords from Python logs; `agent_audit.jsonl` logs `argument_names` only, never values; security-audit metadata is allowlisted to identifiers/counters and drops content-bearing fields | `app/security/sanitizer.py`, `app/security/audit.py`, `app/agents/tool_executor.py`, `app/main.py` | `test_log_redaction_filter`, `test_tool_executor_blocks_unknown_tool_and_logs_without_values`, `test_security_audit_drops_document_content_and_redacts_metadata` |
| **Confidential data in Git** | `.gitignore` excludes `.env`, `/data/`, `/logs/`, `/evaluation/`, local databases, and uploaded documents | `.gitignore`, `.env.example` | Manual inspection |

## 3. Live Local-Model Validation (2026-11-27)

The repeatable `python -m scripts.validate_security_live` check used the installed `llama3.2:3b` model and an isolated temporary security-audit file. It submitted a malicious instruction containing a fake secret and an attempt to invoke an unapproved deletion action.

| Check | Result |
| :--- | :--- |
| Prompt-injection indicators detected | Pass |
| External endpoint rejected | Pass |
| Disallowed tool appeared in the agent trace | No |
| Fake secret or document text leaked into security audit | No |
| Agent request completed | Yes; one tool-trace step |

The check passed. Its scope is intentionally small: it establishes an end-to-end regression signal for the local model and audit path, but it does not establish resilience to a broad injection corpus or prove the absence of all PII leaks.

## 4. Known Residual Risks & Limitations

1. **Authentication & RBAC**: The prototype targets single-host local deployment without multi-user authentication or role-based access control.
2. **Container/OS Sandboxing**: Network restriction is enforced at the Python URL-validation layer (`validate_local_endpoint`); production air-gapped deployments should pair this with OS firewall rules or container network policies.
3. **Model Weight Provenance**: Initial model pulls (`ollama pull` / FastEmbed cache) require verified setup before disconnecting the network.
