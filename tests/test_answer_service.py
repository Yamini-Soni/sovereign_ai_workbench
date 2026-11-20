
import pytest

from app.retrieval import answer_service


@pytest.fixture
def document_metadata():
    return [
        {
            "document_id": "doc_policy",
            "filename": "company_polict.txt",
        },
        {
            "document_id": "doc_operations",
            "filename": "it_operations_policy.txt",
        },
    ]


@pytest.fixture
def annual_leave_chunk():
    return {
        "document_id": "doc_policy",
        "chunk_index": 0,
        "page_number": 1,
        "text": "Every employee receives 20 days of paid annual leave per year.",
        "similarity_score": 0.65,
    }


@pytest.fixture
def remote_work_chunk():
    return {
        "document_id": "doc_operations",
        "chunk_index": 1,
        "page_number": 1,
        "text": "Employees may work remotely up to 2 days per week.",
        "similarity_score": 0.72,
    }


def configure_pipeline(
    monkeypatch,
    document_metadata,
    semantic_chunks,
    all_chunks,
):
    """Replace database and retrieval dependencies with test data."""

    monkeypatch.setattr(
        answer_service,
        "get_all_documents",
        lambda: document_metadata,
    )

    monkeypatch.setattr(
        answer_service,
        "semantic_search",
        lambda **kwargs: semantic_chunks,
    )

    monkeypatch.setattr(
        answer_service,
        "get_all_vectors",
        lambda **kwargs: all_chunks,
    )


def test_answers_facts_from_multiple_documents(
    monkeypatch,
    document_metadata,
    annual_leave_chunk,
    remote_work_chunk,
):
    chunks = [annual_leave_chunk, remote_work_chunk]

    configure_pipeline(
        monkeypatch,
        document_metadata,
        chunks,
        chunks,
    )

    question = (
        "How many days of annual leave do employees receive, "
        "and how many days per week can they work remotely?"
    )

    result = answer_service.answer_question(question)

    assert "20 days" in result["answer"]
    assert "2 days per week" in result["answer"]

    assert result["answer_method"] == (
        "deterministic_policy_evidence_extraction"
    )

    assert len(result["verified_evidence"]) == 2

    filenames = [
        source["filename"]
        for source in result["sources"]
    ]

    assert "company_polict.txt" in filenames
    assert "it_operations_policy.txt" in filenames


def test_full_collection_fallback_finds_missed_evidence(
    monkeypatch,
    document_metadata,
    annual_leave_chunk,
    remote_work_chunk,
):
    # Semantic retrieval returns only the remote-work chunk.
    semantic_chunks = [remote_work_chunk]

    # The complete vector collection contains both chunks.
    all_chunks = [
        {
            key: value
            for key, value in chunk.items()
            if key != "similarity_score"
        }
        for chunk in [annual_leave_chunk, remote_work_chunk]
    ]

    configure_pipeline(
        monkeypatch,
        document_metadata,
        semantic_chunks,
        all_chunks,
    )

    question = (
        "How many days of annual leave do employees receive, "
        "and how many days per week can they work remotely?"
    )

    result = answer_service.answer_question(question)

    assert "20 days" in result["answer"]
    assert "2 days per week" in result["answer"]

    # The annual-leave chunk must be recovered from the full collection.
    sources = result["sources"]

    annual_leave_sources = [
        source
        for source in sources
        if source["filename"] == "company_polict.txt"
    ]

    assert len(annual_leave_sources) == 1

    # This chunk was not returned by semantic retrieval.
    assert annual_leave_sources[0]["similarity_score"] is None


def test_missing_fact_is_not_invented(
    monkeypatch,
    document_metadata,
    annual_leave_chunk,
):
    chunks = [annual_leave_chunk]

    configure_pipeline(
        monkeypatch,
        document_metadata,
        chunks,
        chunks,
    )

    result = answer_service.answer_question(
        "What is the maternity leave duration?"
    )

    assert "could not be found" in result["answer"]
    assert result["verified_evidence"] == []

    assert result["answer_method"] == (
        "deterministic_policy_evidence_extraction"
    )


def test_local_llm_answer_citations_are_formatted(
    monkeypatch,
    document_metadata,
):
    backup_chunk = {
        "document_id": "doc_policy",
        "chunk_index": 0,
        "page_number": 1,
        "text": "Servers are backed up every day at 2 AM.",
        "similarity_score": 0.65,
    }

    configure_pipeline(
        monkeypatch,
        document_metadata,
        [backup_chunk],
        [backup_chunk],
    )

    monkeypatch.setattr(
        answer_service,
        "ask_llm",
        lambda question, chunks: (
            "Servers are backed up every day at 2 AM. "
            "[it_operations_policy.txt | page 1]"
        ),
    )

    result = answer_service.answer_question(
        "How often are servers backed up?"
    )

    assert result["answer_method"] == "local_llm"

    # Backend must use the actual evidence source, not the
    # incorrect filename generated by the model.
    assert (
        "[company_polict.txt | page 1]"
        in result["answer"]
    )

    assert "it_operations_policy.txt" not in result["answer"]
    assert "[Source " not in result["answer"]
def test_local_llm_receives_neighboring_chunks(
    monkeypatch,
    document_metadata,
):
    first_chunk = {
        "document_id": "doc_operations",
        "chunk_index": 0,
        "page_number": 1,
        "text": (
            "Company laptops must be returned "
            "on the employee's last"
        ),
    }

    second_chunk = {
        "document_id": "doc_operations",
        "chunk_index": 1,
        "page_number": 1,
        "text": (
            "must be returned on the employee's "
            "last working day."
        ),
    }

    unrelated_chunk = {
        "document_id": "doc_policy",
        "chunk_index": 1,
        "page_number": 1,
        "text": "The employee's line manager approves leave.",
    }

    configure_pipeline(
        monkeypatch,
        document_metadata,
        [first_chunk],
        [
            first_chunk,
            second_chunk,
            unrelated_chunk,
        ],
    )

    captured_chunks = []

    def fake_ask_llm(question, chunks):
        captured_chunks.extend(chunks)

        return (
            "Company laptops must be returned "
            "on the employee's last working day. "
            "[Source 2 | page 1]"
        )

    monkeypatch.setattr(
        answer_service,
        "ask_llm",
        fake_ask_llm,
    )

    result = answer_service.answer_question(
        "Explain the equipment return policy."
    )

    assert len(captured_chunks) == 2

    assert captured_chunks[0]["chunk_index"] == 0
    assert captured_chunks[1]["chunk_index"] == 1

    assert all(
        chunk["document_id"] == "doc_operations"
        for chunk in captured_chunks
    )

    assert "last working day" in result["answer"]
    assert "[it_operations_policy.txt | page 1]" in result["answer"]

def test_citation_uses_actual_evidence_not_model_filename(
    monkeypatch,
    document_metadata,
):
    incident_chunk = {
        "document_id": "doc_policy",
        "chunk_index": 0,
        "page_number": 1,
        "text": (
            "Security incidents must be reported "
            "to the IT helpdesk within 1 hour."
        ),
    }

    unrelated_chunk = {
        "document_id": "doc_operations",
        "chunk_index": 0,
        "page_number": 1,
        "text": (
            "Employees may work remotely "
            "up to 2 days per week."
        ),
    }

    chunks = [incident_chunk, unrelated_chunk]

    configure_pipeline(
        monkeypatch,
        document_metadata,
        chunks,
        chunks,
    )

    monkeypatch.setattr(
        answer_service,
        "ask_llm",
        lambda question, chunks: (
            "Security incidents must be reported "
            "to the IT helpdesk within 1 hour "
            "[it_operations_policy.txt | page 1]."
        ),
    )

    result = answer_service.answer_question(
        "When must security incidents be reported?"
    )

    assert "within 1 hour" in result["answer"]

    assert (
        "[company_polict.txt | page 1]"
        in result["answer"]
    )

    assert "it_operations_policy.txt" not in result["answer"]

def test_wrong_numeric_claim_does_not_match_evidence_chunk():
    source_chunk = {
        "document_id": "doc_operations",
        "chunk_index": 0,
        "page_number": 1,
        "text": "Employees may work remotely up to 2 days per week.",
    }

    wrong_claim = (
        "Employees may work remotely up to 3 days per week."
    )

    evidence_index = answer_service.find_best_evidence_chunk(
        wrong_claim,
        [source_chunk],
    )

    assert evidence_index is None