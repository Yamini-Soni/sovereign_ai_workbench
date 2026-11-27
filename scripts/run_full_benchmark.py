"""Expanded research benchmark runner for retrieval, QA, agent reliability, latency, and resources."""

import argparse
import json
import math
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

from app.core.config import BASE_DIR, settings
from app.retrieval.answer_service import answer_question
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.semantic_search import semantic_search
from app.storage.model_store import get_model_settings


BENCHMARK_FILE = BASE_DIR / "benchmarks" / "benchmark_cases.json"
OUT_FILE = BASE_DIR / "evaluation" / "full_benchmark_results.json"


def compute_retrieval_metrics(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 5,
) -> dict:
    """Compute Recall@K, Precision@K, and Reciprocal Rank for a single query."""
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k = ranked_doc_ids[:k]
    if not relevant_doc_ids:
        return {
            "recall_at_k": 1.0 if not top_k else 0.0,
            "precision_at_k": 0.0,
            "reciprocal_rank": 0.0,
        }

    hits = [doc_id for doc_id in top_k if doc_id in relevant_doc_ids]
    recall = len(set(hits)) / len(relevant_doc_ids)
    precision = len(hits) / k

    rr = 0.0
    for idx, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_doc_ids:
            rr = 1.0 / idx
            break

    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "reciprocal_rank": round(rr, 4),
    }


def compute_percentile(values: list[float], pct: float) -> float:
    """Compute a linear-interpolated percentile from numeric samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(ordered[low], 4)
    frac = rank - low
    interpolated = ordered[low] + frac * (ordered[high] - ordered[low])
    return round(interpolated, 4)


def evaluate_qa_case(case: dict, answer: str) -> bool:
    """Evaluate answer correctness or refusal behavior using case assertions."""
    answer_lower = (answer or "").lower()
    is_refusal = "could not find" in answer_lower

    if not case["answerable"]:
        return is_refusal

    if is_refusal:
        return False

    has_expected = any(
        term.lower() in answer_lower for term in case.get("expected_any", [])
    )
    has_forbidden = any(
        term.lower() in answer_lower for term in case.get("forbidden", [])
    )
    return has_expected and not has_forbidden


def sample_hardware_metrics() -> dict:
    """Collect current process RAM and optional NVIDIA GPU telemetry."""
    process = psutil.Process()
    mem_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    gpu_info = None

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
        if output:
            parts = [p.strip() for p in output.split(",")]
            if len(parts) >= 4:
                gpu_info = {
                    "gpu_name": parts[0],
                    "vram_used_mib": int(parts[1]),
                    "vram_total_mib": int(parts[2]),
                    "gpu_util_percent": int(parts[3]),
                }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        gpu_info = None

    return {
        "process_rss_mb": mem_mb,
        "system_ram_percent": psutil.virtual_memory().percent,
        "gpu": gpu_info,
    }


def run_benchmark(
    retrieval_mode: str = "semantic",
    top_k: int = 5,
    dry_run: bool = False,
) -> dict:
    """Execute the expanded benchmark suite in live or deterministic dry-run mode."""
    suite = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    qa_cases = suite["qa_cases"]

    latencies: list[float] = []
    qa_results: list[dict] = []
    retrieval_scores: list[dict] = []
    claim_supported_count = 0
    claim_checked_count = 0

    for case in qa_cases:
        t0 = time.perf_counter()
        if dry_run:
            # Deterministic offline baseline check for pipeline validation without Ollama.
            if case["answerable"]:
                stub_token = (case.get("expected_any") or ["verified"])[0]
                answer = f"Verified benchmark response: {stub_token}."
                claim_ver = {"supported": True, "support_ratio": 1.0}
                ranked_docs = [case["doc"]] if case.get("doc") else []
            else:
                answer = "I could not find this in the approved documents."
                claim_ver = {"supported": True, "support_ratio": 1.0}
                ranked_docs = []
            method = "dry_run"
        else:
            res = answer_question(
                question=case["question"],
                retrieval_mode=retrieval_mode,
            )
            answer = res["answer"]
            method = res["answer_method"]
            claim_ver = res.get("claim_verification")
            if retrieval_mode == "hybrid":
                hits = hybrid_search(case["question"], top_k=top_k)
            else:
                hits = semantic_search(case["question"], top_k=top_k)
            ranked_docs = [h["document_id"] for h in hits]

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        passed = evaluate_qa_case(case, answer)
        if claim_ver is not None:
            claim_checked_count += 1
            if claim_ver.get("supported"):
                claim_supported_count += 1

        if case.get("doc") and dry_run:
            r_metrics = compute_retrieval_metrics(ranked_docs, {case["doc"]}, k=top_k)
            retrieval_scores.append(r_metrics)

        qa_results.append({
            "id": case["id"],
            "answerable": case["answerable"],
            "passed": passed,
            "answer_method": method,
            "latency_seconds": round(elapsed, 4),
        })

    answerable = [r for r in qa_results if r["answerable"]]
    unanswerable = [r for r in qa_results if not r["answerable"]]

    report = {
        "benchmark_name": suite["benchmark_name"],
        "mode": "dry_run" if dry_run else "live",
        "retrieval_mode": retrieval_mode,
        "vector_store_backend": settings.VECTOR_STORE_BACKEND,
        "models": get_model_settings(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {
            "total_qa_cases": len(qa_results),
            "answerable_accuracy": round(
                sum(r["passed"] for r in answerable) / len(answerable), 4
            ) if answerable else 0.0,
            "refusal_accuracy": round(
                sum(r["passed"] for r in unanswerable) / len(unanswerable), 4
            ) if unanswerable else 0.0,
            "overall_pass_rate": round(
                sum(r["passed"] for r in qa_results) / len(qa_results), 4
            ) if qa_results else 0.0,
            "claim_support_rate": round(
                claim_supported_count / claim_checked_count, 4
            ) if claim_checked_count else 0.0,
            "median_latency_seconds": round(statistics.median(latencies), 4) if latencies else 0.0,
            "p95_latency_seconds": compute_percentile(latencies, 95),
            "mean_recall_at_k": round(
                statistics.mean(s["recall_at_k"] for s in retrieval_scores), 4
            ) if retrieval_scores else None,
            "mean_mrr_at_k": round(
                statistics.mean(s["reciprocal_rank"] for s in retrieval_scores), 4
            ) if retrieval_scores else None,
        },
        "hardware": sample_hardware_metrics(),
        "results": qa_results,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run expanded Sovereign AI Workbench benchmark.")
    parser.add_argument("--retrieval-mode", choices=["semantic", "hybrid"], default="semantic")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Run deterministic offline harness check.")
    args = parser.parse_args()

    report = run_benchmark(
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        dry_run=args.dry_run,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
