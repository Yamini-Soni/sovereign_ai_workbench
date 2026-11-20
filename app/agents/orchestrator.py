"""Bounded iterative local-LLM multimodal agent orchestrator."""

import ast
import json
import re
import time
import urllib.request

from app.agents.tool_registry import TOOLS
from app.core.config import settings
from app.agents.argument_validator import validate_arguments
from app.storage.model_store import get_model_settings
from app.agents.tool_executor import execute_tool
from app.retrieval.answer_service import answer_question


OLLAMA_URL = "http://localhost:11434/api/chat"


MAX_STEPS = 3
PLANNER_TIMEOUT = 180


# ---------------------------------------------------------
# Number handling and calculation grounding
# ---------------------------------------------------------

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _extract_supported_numbers(question, tool_history):
    """Collect numbers supported by the request or verified evidence."""

    supported = set()

    for match in re.findall(r"\b\d+(?:\.\d+)?\b", question):
        supported.add(float(match))

    words = re.findall(r"\b[a-z]+\b", question.lower())

    for word in words:
        if word in NUMBER_WORDS:
            supported.add(float(NUMBER_WORDS[word]))

    for item in tool_history:
        if item.get("tool") != "answer_document_question":
            continue

        if not item.get("ok"):
            continue

        result = item.get("result") or {}

        for evidence in result.get("verified_evidence") or []:
            value = evidence.get("value")

            try:
                if value is not None:
                    supported.add(float(value))
            except (TypeError, ValueError):
                continue

    return supported


def _validate_calculation_grounding(
    expression,
    question,
    tool_history,
):
    """Reject arithmetic containing unsupported numeric constants."""

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(
            "Invalid mathematical expression."
        ) from error

    supported = _extract_supported_numbers(
        question,
        tool_history,
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError(
                    "Boolean values are not allowed."
                )

            if isinstance(node.value, (int, float)):
                if float(node.value) not in supported:
                    raise ValueError(
                        f"Unsupported calculation number: {node.value}"
                    )


def _requires_calculation(question: str) -> bool:
    """Check whether the user explicitly requested arithmetic."""

    calculation_patterns = [
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bmultiply\b",
        r"\bdivide\b",
        r"\badd\b",
        r"\bsubtract\b",
        r"\btotal\b",
        r"\bsum\b",
        r"\bproduct\b",
        r"\bhow much in total\b",
        r"\bhow many in total\b",
        (
            r"\b(?:over|for|across)\s+"
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
            r"\s+weeks?\b"
        ),
    ]

    question_lower = question.lower()

    return any(
        re.search(pattern, question_lower)
        for pattern in calculation_patterns
    )


def _build_grounded_calculation_expression(
    question,
    tool_history,
):
    """
    Build a multiplication expression only when the user explicitly
    requests a total over a stated number of weeks and RAG has
    verified a per-week value.
    """

    if not _requires_calculation(question):
        return None

    match = re.search(
        r"\b(?:over|for|across)\s+"
        r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
        r"\s+weeks?\b",
        question.lower(),
    )

    if not match:
        return None

    if not re.search(r"\bper\s+week\b", question.lower()):
        return None

    period = match.group(1)

    if period.isdigit():
        weeks = int(period)
    else:
        weeks = NUMBER_WORDS[period]

    if weeks <= 0:
        return None

    for item in tool_history:
        if (
            item.get("tool") != "answer_document_question"
            or not item.get("ok")
        ):
            continue

        result = item.get("result") or {}

        for evidence in result.get("verified_evidence") or []:
            if evidence.get("fact") != "remote_work":
                continue

            try:
                weekly_value = float(evidence["value"])
            except (KeyError, TypeError, ValueError):
                continue

            if weekly_value.is_integer():
                weekly_text = str(int(weekly_value))
            else:
                weekly_text = str(weekly_value)

            return f"{weekly_text} * {weeks}"

    return None


# ---------------------------------------------------------
# Dynamic tool availability
# ---------------------------------------------------------

def _get_available_tools(image_bytes=None) -> dict:
    """Expose vision only when validated image bytes are available."""

    available_tools = {
        name: tool
        for name, tool in TOOLS.items()
        if name != "analyze_image"
    }

    if isinstance(image_bytes, bytes) and image_bytes:
        if "analyze_image" in TOOLS:
            available_tools["analyze_image"] = TOOLS["analyze_image"]

    return available_tools


def _describe_available_tools(available_tools: dict) -> list[dict]:
    """Describe only tools available to the current request."""

    return [
        {
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for name, tool in available_tools.items()
    ]


# ---------------------------------------------------------
# Planner instructions
# ---------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """
You are an iterative tool-selection agent for a secure
enterprise document assistant.

Your responsibility is to complete the user's actual request
using the smallest necessary sequence of approved tools.

You will receive the tools available for the current request.
You MUST select tools only from that list.

DECISION RULES:

1. Return exactly one action per response.
2. The action field must be exactly "tool" or "done".
3. Use only the available tools and their exact argument schemas.
4. Never invent document facts or unsupported numbers.
5. Never answer document questions from memory.
6. For questions about the contents, meaning, summary,
   explanation, or facts of company documents,
   use answer_document_question.

7. Use list_documents ONLY when the user explicitly asks
   to list, show, or identify available documents.

8. Use search_documents only when the user explicitly
   requests relevant passages or targeted retrieval beyond
   answer_document_question.

9. If the user requests a calculation based on a document fact,
   retrieve that fact before calculating.

10. Do not use calculate unless the user requests arithmetic
    or a numerical computation.

11. Do not use search_documents after answer_document_question
    has already answered the user's question.

12. Do not repeat a tool operation that has already succeeded.

13. Read previous tool results carefully before acting.

14. If a successful tool result completely satisfies the
    user's request, return {"action": "done"}.

15. After a successful calculate call, return done if the
    requested calculation is complete.

16. If the user requests multiple calculations or additional
    steps, continue with those requested steps.

17. Do not invent additional tasks or perform unnecessary work.

18. Never exceed the allowed execution limit.

VISION RULES:

19. Use analyze_image only when it appears in the available tools.

20. When the user asks about visible objects, people, scenes,
    image contents, or visual details, use analyze_image.

21. Do not guess visual information from the text question.

22. For a request requiring both image understanding and
    company-document evidence, use both analyze_image and
    answer_document_question.

23. Do not use analyze_image to answer company-policy questions
    that require document evidence.

24. The image is already attached to the current request.
    Never ask for an image path, filename, or base64.

25. The analyze_image tool accepts only its declared question
    argument. Never invent arguments for image bytes.

26. After receiving an image-analysis result, use it as visual
    evidence. Do not repeat image analysis.

IMPORTANT EXAMPLES:

Example A:
User: "How many days of annual leave do employees receive?"
Correct sequence:
- answer_document_question
- done

Example B:
User: "How many days can employees work remotely per week,
and calculate the total over four weeks?"
Correct sequence:
- answer_document_question
- calculate using the verified weekly value and 4
- done

Example C:
User: "How many books are visible in this image?"
Correct sequence:
- analyze_image
- done

Example D:
User: "Compare this image with the company equipment policy."
Correct sequence:
- analyze_image
- answer_document_question
- done

Do not guess document facts or visual details.

TOOL ACTION FORMAT:

{
    "action": "tool",
    "tool": "approved_tool_name",
    "arguments": {}
}

FINISH FORMAT:

{
    "action": "done"
}

You will receive the original request, available tools,
and previous tool results with every planning request.

Return ONLY valid JSON.
"""


# ---------------------------------------------------------
# Local LLM planner
# ---------------------------------------------------------

def plan_with_llm(
    question: str,
    tool_history: list[dict],
    available_tools: dict | None = None,
) -> str:
    """Ask Ollama to select the next single action."""

    if available_tools is None:
        available_tools = _get_available_tools()

    tools_description = json.dumps(
        _describe_available_tools(available_tools),
        ensure_ascii=False,
        indent=2,
    )

    history_description = json.dumps(
        tool_history,
        ensure_ascii=False,
        indent=2,
    )

    user_message = (
        f"Available tools:\n{tools_description}\n\n"
        f"Original user request:\n{question}\n\n"
        f"Previous tool results:\n{history_description}\n\n"
        "Select exactly one next action."
    )

    payload = {
        "model": get_model_settings()["text"],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
        "messages": [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=PLANNER_TIMEOUT,
    ) as response:
        result = json.loads(response.read())

    return result["message"]["content"].strip()


# ---------------------------------------------------------
# Planner action validation
# ---------------------------------------------------------

def parse_action(
    raw_text: str,
    available_tools: dict | None = None,
) -> dict:
    """Parse and validate one proposed planner action."""

    if available_tools is None:
        available_tools = TOOLS

    try:
        action = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            "Planner did not return valid JSON."
        ) from error

    if not isinstance(action, dict):
        raise ValueError(
            "Planner action must be an object."
        )

    action_type = action.get("action")

    if (
        isinstance(action_type, str)
        and action_type in available_tools
        and action.get("tool") == action_type
    ):
        action_type = "tool"

    if action_type == "done":
        if set(action.keys()) != {"action"}:
            raise ValueError(
                "Done action must contain only the action field."
            )

        return {"action": "done"}

    if action_type != "tool":
        raise ValueError(
            "Planner returned an unknown action."
        )

    if set(action.keys()) != {
        "action",
        "tool",
        "arguments",
    }:
        raise ValueError(
            "Tool action must contain action, tool and arguments."
        )

    tool_name = action["tool"]
    arguments = action["arguments"]

    if not isinstance(tool_name, str):
        raise ValueError("Invalid tool name.")

    if tool_name not in available_tools:
        raise ValueError(
            "Planner requested a tool unavailable for this request."
        )

    if not isinstance(arguments, dict):
        raise ValueError(
            "Tool arguments must be an object."
        )

    clean_arguments = validate_arguments(
        parameters=available_tools[tool_name]["parameters"],
        arguments=arguments,
    )

    return {
        "action": "tool",
        "tool": tool_name,
        "arguments": clean_arguments,
    }


# ---------------------------------------------------------
# Question separation for multimodal requests
# ---------------------------------------------------------

def _requires_document_evidence(question: str) -> bool:
    """Identify requests that require company-document evidence."""

    text = question.lower()

    indicators = (
        "according to",
        "company policy",
        "it policy",
        "operations policy",
        "document",
        "policy",
        "company rules",
    )

    return any(term in text for term in indicators)


def _extract_document_question(question: str) -> str:
    """Extract the document-related part of a mixed question."""

    text = question.strip()
    lowered = text.lower()

    marker = "according to"
    index = lowered.find(marker)

    if index >= 0:
        return text[index:].strip()

    return text


def _extract_visual_question(question: str) -> str:
    """Remove the document-related part from a mixed visual request."""

    text = question.strip()

    patterns = (
        r"\balso\s*,?\s*according to\b",
        r"\band\s+according to\b",
        r"\baccording to\b",
        r"\balso\s*,?\s*(?:what|how|where|when|which|who)\b",
    )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            visual_question = text[:match.start()].strip(" ,;.")
            if visual_question:
                return visual_question

    return (
        "Describe the clearly visible objects in the image, "
        "including their colors and positions."
    )


# ---------------------------------------------------------
# Tool result formatting
# ---------------------------------------------------------

def _format_tool_result(
    tool_name: str,
    execution: dict,
) -> str:
    """Format tool output without inventing factual content."""

    if not execution.get("ok"):
        return (
            f"{tool_name} could not complete: "
            f"{execution.get('error', 'Unknown error')}"
        )

    data = execution.get("result") or {}

    if tool_name == "answer_document_question":
        return data.get(
            "answer",
            "No answer returned.",
        )

    if tool_name == "calculate":
        return (
            f"{data['expression']} = {data['result']}"
        )

    if tool_name == "list_documents":
        documents = data.get("documents", [])

        if not documents:
            return "No documents are currently available."

        lines = [
            f"- {document.get('filename', 'Unknown document')}"
            for document in documents
        ]

        return (
            f"Available documents ({len(documents)}):\n"
            + "\n".join(lines)
        )

    if tool_name == "search_documents":
        matches = data.get("matches", [])

        if not matches:
            return "No matching document passages were found."

        lines = []

        for match in matches:
            lines.append(
                f"[{match['filename']} | page {match['page_number']}]\n"
                f"{match['snippet']}"
            )

        return "\n\n".join(lines)

    if tool_name == "analyze_image":
        return (
            "Image analysis result:\n"
            + data.get(
                "description",
                "No visual description returned.",
            )
        )

    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------
# Fallback and error responses
# ---------------------------------------------------------

def _fallback_to_rag(
    question: str,
    start_time: float,
    reason: str,
) -> dict:
    """Use existing RAG if planning fails before execution."""

    result = answer_question(question=question)

    return {
        "question": question,
        "answer": result["answer"],
        "answer_method": result["answer_method"],
        "verified_evidence": result["verified_evidence"],
        "sources": result["sources"],
        "plan_source": "rag_fallback",
        "fallback_reason": reason,
        "tool_results": [],
        "trace": [],
        "completed": True,
        "termination_reason": "rag_fallback",
        "planning_error": reason,
        "latency_seconds": round(
            time.perf_counter() - start_time,
            6,
        ),
    }


def _planning_failure_response(
    question: str,
    start_time: float,
    reason: str,
    tool_results: list,
    trace: list,
) -> dict:
    """Return a safe failure for an image-dependent planning error."""

    return {
        "question": question,
        "answer": (
            "The agent could not complete the request. "
            "Please try again."
        ),
        "plan_source": "local_llm",
        "tool_results": tool_results,
        "trace": trace,
        "completed": False,
        "termination_reason": "planning_error",
        "planning_error": reason,
        "latency_seconds": round(
            time.perf_counter() - start_time,
            6,
        ),
    }


# ---------------------------------------------------------
# Main agent orchestrator
# ---------------------------------------------------------

def run_agent(
    question: str,
    planner=None,
    image_bytes: bytes | None = None,
) -> dict:
    """
    Execute a bounded sequence of approved tools.

    planner:
        Deterministic planner used by tests.

    image_bytes:
        Validated image bytes supplied by the API.
        These bytes are passed only to the vision tool,
        never to the planner.
    """

    if not isinstance(question, str):
        raise ValueError(
            "Question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if image_bytes is not None:
        if (
            not isinstance(image_bytes, bytes)
            or not image_bytes
        ):
            raise ValueError(
                "Image context must contain non-empty validated bytes."
            )

    start_time = time.perf_counter()

    planner_function = planner or plan_with_llm

    available_tools = _get_available_tools(
        image_bytes=image_bytes,
    )

    tool_results = []
    trace = []
    tool_history = []

    completed = False
    termination_reason = None
    planning_error = None

    # -------------------------------------------------
    # 1. Bounded planning and execution loop
    # -------------------------------------------------

    for step_number in range(MAX_STEPS):

        # ---------------------------------------------
        # 1A. Automatic grounded weekly-total calculation
        # ---------------------------------------------

        calculation_already_executed = any(
            item.get("tool") == "calculate"
            for item in tool_history
        )

        if not calculation_already_executed:
            expression = _build_grounded_calculation_expression(
                question,
                tool_history,
            )

            if expression is not None:
                _validate_calculation_grounding(
                    expression,
                    question,
                    tool_history,
                )

                execution = execute_tool(
                    tool_name="calculate",
                    arguments={
                        "expression": expression,
                    },
                )

                tool_results.append(execution)

                trace.append({
                    "step": step_number + 1,
                    "tool": "calculate",
                    "ok": execution["ok"],
                    "argument_names": ["expression"],
                    "duration_seconds": execution.get(
                        "duration_seconds",
                        0.0,
                    ),
                    "error_type": execution.get("error_type"),
                })

                tool_history.append({
                    "tool": "calculate",
                    "ok": execution["ok"],
                    "result": execution.get("result"),
                    "error": execution.get("error"),
                })

                if not execution["ok"]:
                    termination_reason = "calculation_failed"
                    break

                completed = True
                termination_reason = "calculation_completed"
                break

        # ---------------------------------------------
        # 2. Ask the planner for the next action
        # ---------------------------------------------

        try:
            if planner is None:
                raw_action = planner_function(
                    question,
                    tool_history,
                    available_tools,
                )
            else:
                raw_action = planner_function(
                    question,
                    tool_history,
                )

            action = parse_action(
                raw_action,
                available_tools=available_tools,
            )

        except Exception as error:
            planning_error = type(error).__name__

            if not tool_results:
                if image_bytes is None:
                    return _fallback_to_rag(
                        question=question,
                        start_time=start_time,
                        reason=planning_error,
                    )

                return _planning_failure_response(
                    question=question,
                    start_time=start_time,
                    reason=planning_error,
                    tool_results=tool_results,
                    trace=trace,
                )

            termination_reason = "planning_error"
            break

        # ---------------------------------------------
        # 3. Handle planner completion
        # ---------------------------------------------

        if action["action"] == "done":

            if not tool_results:
                if image_bytes is None:
                    return _fallback_to_rag(
                        question=question,
                        start_time=start_time,
                        reason="NoToolsSelected",
                    )

                return _planning_failure_response(
                    question=question,
                    start_time=start_time,
                    reason="NoToolsSelected",
                    tool_results=tool_results,
                    trace=trace,
                )

            completed = True
            termination_reason = "planner_done"
            break

        tool_name = action["tool"]
        arguments = action["arguments"]

        # ---------------------------------------------
        # 4. Prevent repeated successful image analysis
        # ---------------------------------------------

        if (
            tool_name == "analyze_image"
            and any(
                item.get("tool") == "analyze_image"
                and item.get("ok")
                for item in tool_history
            )
        ):
            completed = True
            termination_reason = "duplicate_tool_prevented"
            break

        # ---------------------------------------------
        # 5. Prevent redundant document retrieval
        # ---------------------------------------------

        document_already_retrieved = any(
            item.get("tool") == "answer_document_question"
            and item.get("ok")
            for item in tool_history
        )

        if (
            tool_name == "answer_document_question"
            and document_already_retrieved
        ):
            completed = True
            termination_reason = "document_already_retrieved"
            break

        # ---------------------------------------------
        # 6. Validate calculation grounding
        # ---------------------------------------------

        if tool_name == "calculate":
            expression = arguments["expression"]

            try:
                _validate_calculation_grounding(
                    expression=expression,
                    question=question,
                    tool_history=tool_history,
                )

            except ValueError as grounding_error:

                has_document_evidence = any(
                    item.get("tool") == "answer_document_question"
                    and item.get("ok")
                    for item in tool_history
                )

                if not has_document_evidence:
                    execution = execute_tool(
                        tool_name="answer_document_question",
                        arguments={
                            "question": _extract_document_question(
                                question
                            ),
                        },
                    )

                    tool_results.append(execution)

                    trace.append({
                        "step": step_number + 1,
                        "tool": "answer_document_question",
                        "ok": execution["ok"],
                        "argument_names": ["question"],
                        "duration_seconds": execution.get(
                            "duration_seconds",
                            0.0,
                        ),
                        "error_type": execution.get("error_type"),
                        "policy_override": True,
                    })

                    tool_history.append({
                        "tool": "answer_document_question",
                        "ok": execution["ok"],
                        "result": execution.get("result"),
                        "error": execution.get("error"),
                    })

                    if not execution["ok"]:
                        termination_reason = "document_retrieval_failed"
                        break

                    continue

                rejected_execution = {
                    "tool": "calculate",
                    "ok": False,
                    "error": str(grounding_error),
                    "error_type": "GroundingError",
                    "duration_seconds": 0.0,
                }

                tool_results.append(rejected_execution)

                trace.append({
                    "step": step_number + 1,
                    "tool": "calculate",
                    "ok": False,
                    "argument_names": ["expression"],
                    "duration_seconds": 0.0,
                    "error_type": "GroundingError",
                })

                termination_reason = "grounding_rejected"
                break

        # ---------------------------------------------
        # 7. Prevent redundant retrieval after calculation
        # ---------------------------------------------

        calculation_completed = any(
            item.get("tool") == "calculate"
            and item.get("ok")
            for item in tool_history
        )

        document_retrieval_completed = any(
            item.get("tool") == "answer_document_question"
            and item.get("ok")
            for item in tool_history
        )

        if (
            tool_name == "answer_document_question"
            and calculation_completed
            and document_retrieval_completed
            and _requires_calculation(question)
        ):
            completed = True
            termination_reason = "calculation_completed"
            break

        # ---------------------------------------------
        # 8. Execute the approved tool
        # ---------------------------------------------

        if tool_name == "analyze_image":
            image_arguments = arguments

            if _requires_document_evidence(question):
                image_arguments = {
                    **arguments,
                    "question": _extract_visual_question(question),
                }

            execution = execute_tool(
                tool_name=tool_name,
                arguments=image_arguments,
                image_bytes=image_bytes,
            )

        else:
            execution = execute_tool(
                tool_name=tool_name,
                arguments=arguments,
            )

        tool_results.append(execution)

        # ---------------------------------------------
        # 9. Record execution trace
        # ---------------------------------------------

        trace.append({
            "step": step_number + 1,
            "tool": tool_name,
            "ok": execution["ok"],
            "argument_names": sorted(arguments.keys()),
            "duration_seconds": execution.get(
                "duration_seconds",
                0.0,
            ),
            "error_type": execution.get("error_type"),
        })

        # ---------------------------------------------
        # 10. Update tool history
        # ---------------------------------------------

        tool_history.append({
            "tool": tool_name,
            "ok": execution["ok"],
            "result": execution.get("result"),
            "error": execution.get("error"),
        })

        # ---------------------------------------------
        # 11. Handle mixed image + document requests
        # ---------------------------------------------

        if (
            tool_name == "analyze_image"
            and execution["ok"]
            and _requires_document_evidence(question)
        ):
            document_already_retrieved = any(
                item.get("tool") == "answer_document_question"
                and item.get("ok")
                for item in tool_history
            )

            if not document_already_retrieved:
                document_execution = execute_tool(
                    tool_name="answer_document_question",
                    arguments={
                        "question": _extract_document_question(
                            question
                        ),
                    },
                )

                tool_results.append(document_execution)

                trace.append({
                    "step": step_number + 2,
                    "tool": "answer_document_question",
                    "ok": document_execution["ok"],
                    "argument_names": ["question"],
                    "duration_seconds": document_execution.get(
                        "duration_seconds",
                        0.0,
                    ),
                    "error_type": document_execution.get("error_type"),
                    "policy_override": True,
                })

                tool_history.append({
                    "tool": "answer_document_question",
                    "ok": document_execution["ok"],
                    "result": document_execution.get("result"),
                    "error": document_execution.get("error"),
                })

                completed = document_execution["ok"]

                termination_reason = (
                    "multimodal_request_completed"
                    if completed
                    else "document_retrieval_failed"
                )

                break

        # ---------------------------------------------
        # 12. Stop after failed tool execution
        # ---------------------------------------------

        if not execution["ok"]:
            termination_reason = "tool_execution_failed"
            break

        # ---------------------------------------------
        # 13. Early completion for ordinary document questions
        # ---------------------------------------------

        if (
            planner is None
            and image_bytes is None
            and tool_name == "answer_document_question"
            and not _requires_calculation(question)
        ):
            completed = True
            termination_reason = "document_answered"
            break

        # The planner decides whether another action is needed
        # after other successful tool executions.

    # -------------------------------------------------
    # 14. Handle maximum execution limit
    # -------------------------------------------------

    if not completed and termination_reason is None:
        termination_reason = "max_steps_reached"

    # -------------------------------------------------
    # 15. Format final response
    # -------------------------------------------------

    formatted_results = []

    for execution in tool_results:
        formatted_results.append(
            _format_tool_result(
                execution["tool"],
                execution,
            )
        )

    answer = "\n\n".join(formatted_results)

    # -------------------------------------------------
    # 16. Return final agent response
    # -------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "plan_source": (
            "local_llm"
            if planner is None
            else "custom_planner"
        ),
        "tool_results": tool_results,
        "trace": trace,
        "completed": completed,
        "termination_reason": termination_reason,
        "planning_error": planning_error,
        "latency_seconds": round(
            time.perf_counter() - start_time,
            6,
        ),
    }