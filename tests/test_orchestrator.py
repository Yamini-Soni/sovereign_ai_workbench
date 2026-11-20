
import json

import pytest

from app.agents import orchestrator


def make_action(tool_name, arguments):
    """Create a valid planner tool action."""
    return json.dumps({
        "action": "tool",
        "tool": tool_name,
        "arguments": arguments,
    })


def make_done():
    """Create a valid planner completion action."""
    return json.dumps({"action": "done"})


def test_single_tool_execution_and_done(monkeypatch):
    actions = iter([
        make_action(
            "answer_document_question",
            {"question": "How many annual leave days?"},
        ),
        make_done(),
    ])

    executed_tools = []

    def mock_planner(question, tool_history):
        return next(actions)

    def mock_execute_tool(tool_name, arguments):
        executed_tools.append(tool_name)

        return {
            "ok": True,
            "tool": tool_name,
            "result": {
                "answer": (
                    "Employees receive 20 days of annual leave per year."
                ),
                "verified_evidence": [
                    {
                        "fact": "annual_leave",
                        "value": "20",
                        "unit": "days per year",
                    }
                ],
            },
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr(
        orchestrator,
        "execute_tool",
        mock_execute_tool,
    )

    result = orchestrator.run_agent(
        "How many annual leave days?",
        planner=mock_planner,
    )

    assert "20 days" in result["answer"]
    assert result["completed"] is True
    assert result["termination_reason"] == "planner_done"
    assert executed_tools == ["answer_document_question"]


def test_second_tool_receives_first_tool_result(monkeypatch):
    actions = iter([
        make_action(
            "answer_document_question",
            {
                "question": (
                    "How many days per week can employees work remotely?"
                )
            },
        ),
        make_action(
            "calculate",
            {"expression": "2 * 4"},
        ),
        make_done(),
    ])

    planner_calls = []
    executed_tools = []

    def mock_planner(question, tool_history):
        planner_calls.append({
            "question": question,
            "history": list(tool_history),
        })

        return next(actions)

    def mock_execute_tool(tool_name, arguments):
        executed_tools.append(tool_name)

        if tool_name == "answer_document_question":
            return {
                "ok": True,
                "tool": tool_name,
                "result": {
                    "answer": (
                        "Employees can work remotely up to 2 days per week."
                    ),
                    "verified_evidence": [
                        {
                            "fact": "remote_work",
                            "value": "2",
                            "unit": "days per week",
                        }
                    ],
                },
                "duration_seconds": 0.01,
            }

        if tool_name == "calculate":
            return {
                "ok": True,
                "tool": tool_name,
                "result": {
                    "expression": "2 * 4",
                    "result": 8,
                },
                "duration_seconds": 0.01,
            }

        raise AssertionError("Unexpected tool execution")

    monkeypatch.setattr(
        orchestrator,
        "execute_tool",
        mock_execute_tool,
    )

    result = orchestrator.run_agent(
        "Find remote-work days and calculate the total over four weeks.",
        planner=mock_planner,
    )

    assert len(planner_calls) == 3

    # First planning turn has no previous results.
    assert planner_calls[0]["history"] == []

    # Second planning turn receives the document answer and evidence.
    second_history = planner_calls[1]["history"]

    assert len(second_history) == 1
    assert second_history[0]["tool"] == "answer_document_question"
    assert second_history[0]["ok"] is True

    evidence = second_history[0]["result"]["verified_evidence"]
    assert evidence[0]["value"] == "2"

    # Third planning turn receives both tool results.
    third_history = planner_calls[2]["history"]

    assert len(third_history) == 2
    assert third_history[1]["tool"] == "calculate"

    assert executed_tools == [
        "answer_document_question",
        "calculate",
    ]

    assert "2 * 4 = 8" in result["answer"]
    assert result["completed"] is True
    assert result["termination_reason"] == "planner_done"


def test_agent_stops_after_maximum_steps(monkeypatch):
    actions = iter([
        make_action("calculate", {"expression": "1 + 1"}),
        make_action("calculate", {"expression": "2 + 2"}),
        make_action("calculate", {"expression": "3 + 3"}),
    ])

    execution_count = 0

    def mock_planner(question, tool_history):
        return next(actions)

    def mock_execute_tool(tool_name, arguments):
        nonlocal execution_count
        execution_count += 1

        return {
            "ok": True,
            "tool": tool_name,
            "result": {
                "expression": arguments["expression"],
                "result": execution_count * 2,
            },
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr(
        orchestrator,
        "execute_tool",
        mock_execute_tool,
    )

    result = orchestrator.run_agent(
        "Perform three calculations using 1, 2, and 3.",
        planner=mock_planner,
    )

    assert execution_count == orchestrator.MAX_STEPS
    assert len(result["tool_results"]) == orchestrator.MAX_STEPS
    assert result["completed"] is False
    assert result["termination_reason"] == "max_steps_reached"


def test_unapproved_tool_is_never_executed(monkeypatch):
    executed_tools = []
    fallback_calls = []

    def mock_planner(question, tool_history):
        return json.dumps({
            "action": "tool",
            "tool": "delete_documents",
            "arguments": {},
        })

    def mock_execute_tool(tool_name, arguments):
        executed_tools.append(tool_name)
        raise AssertionError("Unapproved tool must never execute")

    def mock_fallback(question, start_time, reason):
        fallback_calls.append(reason)

        return {
            "question": question,
            "answer": "Fallback answer.",
            "plan_source": "rag_fallback",
            "tool_results": [],
            "trace": [],
            "completed": True,
            "termination_reason": "rag_fallback",
        }

    monkeypatch.setattr(
        orchestrator,
        "execute_tool",
        mock_execute_tool,
    )

    monkeypatch.setattr(
        orchestrator,
        "_fallback_to_rag",
        mock_fallback,
    )

    result = orchestrator.run_agent(
        "Delete all documents.",
        planner=mock_planner,
    )

    assert executed_tools == []
    assert fallback_calls == ["ValueError"]
    assert result["plan_source"] == "rag_fallback"


def test_planner_failure_uses_rag_fallback(monkeypatch):
    fallback_calls = []

    def mock_planner(question, tool_history):
        raise RuntimeError("Simulated planner failure")

    def mock_fallback(question, start_time, reason):
        fallback_calls.append({
            "question": question,
            "reason": reason,
        })

        return {
            "question": question,
            "answer": "Fallback answer.",
            "plan_source": "rag_fallback",
            "tool_results": [],
            "trace": [],
            "completed": True,
            "termination_reason": "rag_fallback",
        }

    monkeypatch.setattr(
        orchestrator,
        "_fallback_to_rag",
        mock_fallback,
    )

    result = orchestrator.run_agent(
        "How many annual leave days?",
        planner=mock_planner,
    )

    assert result["plan_source"] == "rag_fallback"
    assert result["answer"] == "Fallback answer."
    assert fallback_calls[0]["reason"] == "RuntimeError"


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        orchestrator.run_agent("   ")


def test_unsupported_calculation_forces_document_retrieval(monkeypatch):
    actions = iter([
        # The planner initially proposes an unsupported calculation.
        make_action(
            "calculate",
            {"expression": "2 * 5"},
        ),

        # After retrieval, it uses the actual policy value.
        make_action(
            "calculate",
            {"expression": "2 * 4"},
        ),

        make_done(),
    ])

    executed_tools = []
    planner_calls = []

    def mock_planner(question, tool_history):
        planner_calls.append(list(tool_history))
        return next(actions)

    def mock_execute_tool(tool_name, arguments):
        executed_tools.append(tool_name)

        if tool_name == "answer_document_question":
            return {
                "ok": True,
                "tool": tool_name,
                "result": {
                    "answer": (
                        "Employees can work remotely up to "
                        "2 days per week."
                    ),
                    "verified_evidence": [
                        {
                            "fact": "remote_work",
                            "value": "2",
                            "unit": "days per week",
                        }
                    ],
                },
                "duration_seconds": 0.01,
            }

        if tool_name == "calculate":
            expression = arguments["expression"]

            return {
                "ok": True,
                "tool": tool_name,
                "result": {
                    "expression": expression,
                    "result": 8,
                },
                "duration_seconds": 0.01,
            }

        raise AssertionError(f"Unexpected tool: {tool_name}")

    monkeypatch.setattr(
        orchestrator,
        "execute_tool",
        mock_execute_tool,
    )

    result = orchestrator.run_agent(
        "Calculate total remote-work days over four weeks.",
        planner=mock_planner,
    )

    # Retrieval must happen before calculation.
    assert executed_tools == [
        "answer_document_question",
        "calculate",
    ]

    # Three planner calls: retrieve, calculate, finish.
    assert len(planner_calls) == 3

    # The second planning turn must receive retrieved evidence.
    assert planner_calls[1][0]["tool"] == "answer_document_question"

    evidence = planner_calls[1][0]["result"]["verified_evidence"]
    assert evidence[0]["value"] == "2"

    # The third planning turn receives the calculation result.
    assert len(planner_calls[2]) == 2
    assert planner_calls[2][1]["tool"] == "calculate"

    assert "2 * 4 = 8" in result["answer"]
    assert result["completed"] is True
    assert result["termination_reason"] == "planner_done"

def test_repeated_image_analysis_is_blocked(monkeypatch):
    actions = iter([
        make_action(
            "analyze_image",
            {"question": "Describe the objects."},
        ),
        make_action(
            "analyze_image",
            {"question": "Describe the objects again."},
        ),
    ])

    executed_tools = []

    def mock_planner(question, tool_history):
        return next(actions)

    def mock_execute_tool(
        tool_name,
        arguments,
        image_bytes=None,
    ):
        executed_tools.append(tool_name)

        return {
            "ok": True,
            "tool": tool_name,
            "result": {
                "description": "Three books and a pen."
            },
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr(
        orchestrator,
        "execute_tool",
        mock_execute_tool,
    )

    result = orchestrator.run_agent(
        "Describe the objects in this image.",
        planner=mock_planner,
        image_bytes=b"\xff\xd8\xfftest",
    )

    assert executed_tools == ["analyze_image"]
    assert len(result["tool_results"]) == 1
    assert result["completed"] is True
    assert result["termination_reason"] == "duplicate_tool_prevented"