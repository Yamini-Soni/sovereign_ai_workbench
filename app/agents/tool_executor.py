
"""Controlled execution of approved agent tools."""

import json
import time

from datetime import datetime, timezone

from app.core.config import BASE_DIR
from app.agents.argument_validator import (
    validate_arguments,
    ValidationError,
)
from app.agents.tool_registry import TOOLS


# Audit log location.
AUDIT_FILE = BASE_DIR / "logs" / "agent_audit.jsonl"


# ---------------------------------------------------------
# Audit logging
# ---------------------------------------------------------

def _write_audit_event(event: dict) -> None:
    """Append one audit event to the local JSONL audit log."""

    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with AUDIT_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(event, ensure_ascii=False) + "\n"
        )
        file.flush()


def _create_audit_event(
    tool_name: str,
    status: str,
    argument_names: list,
    duration_seconds: float = 0.0,
    error_type: str | None = None,
) -> dict:
    """Create an audit record without storing sensitive values."""

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "status": status,
        "argument_names": argument_names,
        "duration_seconds": round(duration_seconds, 6),
        "error_type": error_type,
    }


# ---------------------------------------------------------
# Controlled tool execution
# ---------------------------------------------------------

def execute_tool(
    tool_name: str,
    arguments,
    image_bytes: bytes | None = None,
) -> dict:
    """
    Execute an approved tool after validating its arguments.

    Image bytes are trusted request context. They are never
    accepted from planner-generated arguments.

    Only the analyze_image tool receives image bytes.
    """

    start_time = time.perf_counter()

    # Record only planner argument names, never their values.
    argument_names = (
        sorted(arguments.keys())
        if isinstance(arguments, dict)
        else []
    )

    # 1. Check whether the requested tool is approved.
    if tool_name not in TOOLS:

        event = _create_audit_event(
            tool_name=tool_name,
            status="rejected",
            argument_names=argument_names,
            error_type="UnknownTool",
        )

        _write_audit_event(event)

        return {
            "ok": False,
            "tool": tool_name,
            "error": "Tool is not approved.",
        }

    tool = TOOLS[tool_name]

    # 2. Validate planner-generated arguments.
    try:
        clean_arguments = validate_arguments(
            parameters=tool["parameters"],
            arguments=arguments,
        )

    except ValidationError as error:

        event = _create_audit_event(
            tool_name=tool_name,
            status="rejected",
            argument_names=argument_names,
            duration_seconds=time.perf_counter() - start_time,
            error_type=type(error).__name__,
        )

        _write_audit_event(event)

        return {
            "ok": False,
            "tool": tool_name,
            "error": str(error),
        }

    # 3. Validate trusted image context separately.
    if tool_name == "analyze_image":

        if (
            not isinstance(image_bytes, bytes)
            or not image_bytes
        ):

            event = _create_audit_event(
                tool_name=tool_name,
                status="rejected",
                argument_names=sorted(clean_arguments.keys()),
                duration_seconds=time.perf_counter() - start_time,
                error_type="MissingImageContext",
            )

            _write_audit_event(event)

            return {
                "ok": False,
                "tool": tool_name,
                "error": "A validated image is required for image analysis.",
            }

    # 4. Record that execution is about to start.
    # If auditing fails, the tool must not execute.
    start_event = _create_audit_event(
        tool_name=tool_name,
        status="started",
        argument_names=sorted(clean_arguments.keys()),
    )

    _write_audit_event(start_event)

    # 5. Execute the registered Python function.
    try:

        if tool_name == "analyze_image":

            # Inject trusted image bytes separately.
            # The planner cannot supply or modify these bytes.
            result = tool["function"](
                **clean_arguments,
                image_bytes=image_bytes,
            )

        else:

            # All other tools receive only validated arguments.
            result = tool["function"](**clean_arguments)

    except Exception as error:

        duration = time.perf_counter() - start_time

        failure_event = _create_audit_event(
            tool_name=tool_name,
            status="failed",
            argument_names=sorted(clean_arguments.keys()),
            duration_seconds=duration,
            error_type=type(error).__name__,
        )

        _write_audit_event(failure_event)

        return {
            "ok": False,
            "tool": tool_name,
            "error": "Tool execution failed.",
            "error_type": type(error).__name__,
        }

    # 6. Record successful completion.
    duration = time.perf_counter() - start_time

    completion_event = _create_audit_event(
        tool_name=tool_name,
        status="completed",
        argument_names=sorted(clean_arguments.keys()),
        duration_seconds=duration,
    )

    _write_audit_event(completion_event)

    # 7. Return the result to the orchestrator.
    return {
        "ok": True,
        "tool": tool_name,
        "result": result,
        "duration_seconds": round(duration, 6),
    }
