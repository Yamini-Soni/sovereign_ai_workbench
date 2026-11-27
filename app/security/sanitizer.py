"""Input sanitization, path-traversal guards, prompt-injection detection, and log redaction."""

import logging
import re
from pathlib import Path


_DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:prior|previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"override\s+(?:system|safety|security)\s+(?:prompt|rules|instructions)", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"\b(?:execute|run|invoke|call)\s+(?:shell|bash|powershell|cmd|os\.system|subprocess|delete_documents)\b", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
]

_SENSITIVE_PATTERNS = [
    (re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|authorization)\s*[:=]\s*(?:Bearer\s+)?([^\s,;\"']+)"), r"\1=[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "[REDACTED_API_KEY]"),
]


def sanitize_filename(filename: str | None) -> str:
    """
    Strip directory components and reject null bytes or path-traversal names.

    Returns a single safe filename component.
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Filename is required.")

    if "\x00" in filename:
        raise ValueError("Filename contains invalid null bytes.")

    # Normalize Windows and POSIX separators before extracting basename.
    normalized = filename.replace("\\", "/")
    basename = Path(normalized).name.strip()

    if not basename or basename in {".", ".."}:
        raise ValueError("Invalid filename.")

    return basename


def validate_document_id(document_id: str) -> str:
    """Ensure a document ID is strictly a safe single identifier without path traversal."""
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    candidate = document_id.strip()

    if not _DOCUMENT_ID_PATTERN.match(candidate):
        raise ValueError("Invalid document ID format.")

    return candidate


def resolve_safe_path(base_dir: Path, relative_or_child: str | Path) -> Path:
    """Resolve a path and enforce that it stays strictly inside base_dir."""
    resolved_base = base_dir.resolve()
    candidate = (resolved_base / relative_or_child).resolve()

    if not candidate.is_relative_to(resolved_base):
        raise ValueError("Path escapes allowed storage directory.")

    return candidate


def detect_prompt_injection(text: str) -> list[str]:
    """Return matched prompt-injection indicator descriptions found in untrusted text."""
    if not text or not isinstance(text, str):
        return []

    findings = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(match.group(0))

    return findings


def sanitize_retrieved_context(text: str) -> tuple[str, list[str]]:
    """
    Detect and neutralize prompt-injection directives inside retrieved document chunks.

    Preserves legitimate document facts while stripping control-tag impersonation
    and flagging instruction-override phrases.
    """
    if not text:
        return "", []

    findings = detect_prompt_injection(text)
    cleaned = re.sub(r"<\s*/?\s*system\s*>", "[tag_removed]", text, flags=re.IGNORECASE)
    return cleaned, findings


def redact_sensitive_text(text: str) -> str:
    """Mask potential credentials, bearer tokens, and secrets before logging."""
    if not text or not isinstance(text, str):
        return ""

    redacted = text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class SensitiveDataFilter(logging.Filter):
    """Logging filter that redacts credentials and tokens from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_text(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_sensitive_text(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: redact_sensitive_text(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True
