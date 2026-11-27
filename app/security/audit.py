"""Security event audit logging without persisting sensitive content."""

import json
from datetime import datetime, timezone

from app.core.config import BASE_DIR
from app.security.sanitizer import redact_sensitive_text


SECURITY_AUDIT_FILE = BASE_DIR / "logs" / "security_audit.jsonl"


def log_security_event(
    event_type: str,
    status: str,
    detail: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Append a redacted security event record to logs/security_audit.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "status": status,
        "detail": redact_sensitive_text(detail) if detail else None,
        "metadata": metadata or {},
    }

    SECURITY_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SECURITY_AUDIT_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    return record


def read_security_audit_summary() -> dict:
    """Summarize recorded security and agent audit events."""
    agent_audit = BASE_DIR / "logs" / "agent_audit.jsonl"

    total = 0
    rejected = 0
    by_type: dict[str, int] = {}

    for path in (SECURITY_AUDIT_FILE, agent_audit):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                status = item.get("status", "")
                if status in {"rejected", "blocked", "failed"}:
                    rejected += 1
                key = item.get("event_type") or item.get("tool") or "unknown"
                by_type[key] = by_type.get(key, 0) + 1

    return {
        "audit_events_total": total,
        "rejected_events_total": rejected,
        "events_by_type": by_type,
    }
