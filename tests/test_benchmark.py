"""Tests for expanded benchmark metrics and offline dry-run execution."""

from scripts.run_full_benchmark import (
    compute_percentile,
    compute_retrieval_metrics,
    evaluate_qa_case,
    run_benchmark,
)


def test_retrieval_metrics_computation():
    metrics = compute_retrieval_metrics(
        ranked_doc_ids=["doc_other", "doc_target", "doc_third"],
        relevant_doc_ids={"doc_target"},
        k=3,
    )
    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == round(1 / 3, 4)
    assert metrics["reciprocal_rank"] == 0.5


def test_percentile_calculation():
    assert compute_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == 4.8
    assert compute_percentile([], 95) == 0.0


def test_evaluate_qa_case_handles_contradiction_and_refusal():
    q05 = {
        "answerable": True,
        "expected_any": ["12"],
        "forbidden": ["not explicitly stated"],
    }
    assert evaluate_qa_case(q05, "Passwords require at least 12 characters.") is True
    assert evaluate_qa_case(q05, "Minimum length is not explicitly stated, maybe 12.") is False

    unanswerable = {"answerable": False, "expected_any": [], "forbidden": []}
    assert evaluate_qa_case(unanswerable, "I could not find this in the approved documents.") is True
    assert evaluate_qa_case(unanswerable, "The CEO earns $500,000.") is False


def test_benchmark_dry_run_completes_all_cases():
    report = run_benchmark(dry_run=True)
    summary = report["summary"]
    assert summary["total_qa_cases"] == 26
    assert summary["answerable_accuracy"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["overall_pass_rate"] == 1.0
    assert report["hardware"]["process_rss_mb"] > 0
