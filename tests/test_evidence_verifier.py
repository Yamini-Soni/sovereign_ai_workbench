
import pytest

from app.retrieval.evidence_verifier import (
    verify_policy_facts,
    verify_password_length,
)


@pytest.fixture
def policy_chunks():
    return [
        {
            "text": (
                "Every employee receives 20 days of paid annual leave "
                "per year. "
                "Employees may work remotely up to 2 days per week."
            ),
            "filename": "company_policy.txt",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "text": (
                "Eligible employees receive 26 weeks of maternity leave."
            ),
            "filename": "hr_policy.txt",
            "page_number": 2,
            "chunk_index": 1,
        },
    ]


@pytest.fixture
def password_chunks():
    return [
        {
            "text": (
                "Passwords must contain at least 12 characters. "
                "Passwords must be changed every 90 days."
            ),
            "filename": "company_policy.txt",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]


def test_extracts_annual_leave(policy_chunks):
    result = verify_policy_facts(
        "How many days of annual leave do employees receive?",
        policy_chunks,
    )

    assert result is not None
    assert "20 days" in result["answer"]
    assert result["evidence"][0]["fact"] == "annual_leave"
    assert result["evidence"][0]["filename"] == "company_policy.txt"
    assert result["evidence"][0]["page_number"] == 1


def test_extracts_remote_work(policy_chunks):
    result = verify_policy_facts(
        "How many days per week can employees work remotely?",
        policy_chunks,
    )

    assert result is not None
    assert "2 days per week" in result["answer"]
    assert result["evidence"][0]["fact"] == "remote_work"


def test_extracts_multiple_policy_facts(policy_chunks):
    result = verify_policy_facts(
        "How many days of annual leave do employees receive, "
        "and how many days per week can they work remotely?",
        policy_chunks,
    )

    assert result is not None
    assert len(result["evidence"]) == 2

    facts = [item["fact"] for item in result["evidence"]]

    assert "annual_leave" in facts
    assert "remote_work" in facts

    assert "20 days" in result["answer"]
    assert "2 days per week" in result["answer"]


def test_extracts_maternity_leave(policy_chunks):
    result = verify_policy_facts(
        "What is the maternity leave duration?",
        policy_chunks,
    )

    assert result is not None
    assert "26 weeks" in result["answer"]
    assert result["evidence"][0]["fact"] == "maternity_leave"
    assert result["evidence"][0]["filename"] == "hr_policy.txt"


def test_reports_missing_policy_fact():
    chunks = [
        {
            "text": "Employees receive one company laptop.",
            "filename": "equipment_policy.txt",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]

    result = verify_policy_facts(
        "What is the maternity leave duration?",
        chunks,
    )

    assert result is not None
    assert "could not be found" in result["answer"]
    assert result["evidence"] == []


def test_returns_none_for_unrelated_policy_question(policy_chunks):
    result = verify_policy_facts(
        "How often are servers backed up?",
        policy_chunks,
    )

    assert result is None


def test_extracts_password_minimum_length(password_chunks):
    result = verify_password_length(
        "What is the minimum password length?",
        password_chunks,
    )

    assert result is not None
    assert "12 characters" in result["answer"]
    assert result["filename"] == "company_policy.txt"
    assert result["page_number"] == 1
    assert result["method"] == "deterministic_evidence_extraction"


def test_reports_missing_password_requirement():
    chunks = [
        {
            "text": "Passwords must be changed regularly.",
            "filename": "security_policy.txt",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]

    result = verify_password_length(
        "What is the minimum password length?",
        chunks,
    )

    assert result is None


def test_returns_none_for_unrelated_password_question(password_chunks):
    result = verify_password_length(
        "How often must passwords be changed?",
        password_chunks,
    )

    assert result is None
