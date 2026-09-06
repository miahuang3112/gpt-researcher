import hashlib
import json
from pathlib import Path

from evals.academic_quality.measure_score_reliability import (
    _safe_error,
    _stability,
    build_reliability_report,
    build_heldout_repeat_stability,
    load_frozen_selections,
)


ROOT = Path(__file__).resolve().parents[2]


def test_safe_error_classifies_unreachable_ollama_without_message() -> None:
    error = RuntimeError(
        "Could not connect to Ollama at http://127.0.0.1:11434/private."
    )

    assert _safe_error(error) == {
        "type": "RuntimeError",
        "code": "ollama_unreachable",
    }


def test_report_separates_failed_calls_from_retry_archives() -> None:
    report = build_reliability_report(
        records=[_record("paper-1", "novelty", 1, 4)],
        reference_by_key={("paper-1", "novelty"): 4},
        run_count=1,
        metadata={"model": "test"},
        attempt_failures=[
            {
                "paper_id": "paper-1",
                "dimension": "novelty",
                "run": 1,
                "attempt": 1,
                "error": {"code": "invalid_json"},
            },
            {
                "paper_id": "paper-1",
                "dimension": "novelty",
                "run": 1,
                "error": {"code": "invalid_json"},
                "superseded_for_retry": True,
            },
        ],
    )

    assert report["model_call_count"] == 2
    assert report["failed_model_call_count"] == 1
    assert report["recovered_item_count"] == 1
    assert report["superseded_terminal_record_count"] == 1


def _record(paper: str, dimension: str, run: int, score: float) -> dict:
    return {
        "paper_id": paper,
        "dimension": dimension,
        "run": run,
        "evaluation": {"status": "scored", "score": score},
        "error": None,
    }


def test_loads_exact_five_paper_frozen_hybrid_selections() -> None:
    selections, metadata = load_frozen_selections(
        ROOT / "evals/academic_retrieval/results/ret09-development-tuning-qwen3-embedding-4b.json",
        ROOT / "evals/academic_retrieval/results/ret10b-heldout-test-qwen3-embedding-4b.json",
    )
    assert len(selections) == 45
    assert {paper for paper, _ in selections} == {
        "arxiv:1706.03762",
        "arxiv:1810.04805",
        "arxiv:2005.11401",
        "arxiv:2106.09685",
        "arxiv:2410.02902",
    }
    assert all(len(chunk_ids) == 4 for chunk_ids in selections.values())
    assert metadata["heldout_used_for_retuning"] is False


def test_stability_reports_exact_agreement_and_pairwise_correlation() -> None:
    summary = _stability([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert summary["exact_run_agreement_rate"] == 1.0
    assert summary["mean_within_item_standard_deviation"] == 0.0
    assert summary["mean_pairwise_run_pearson"] == 1.0
    assert summary["mean_pairwise_run_spearman"] == 1.0


def test_report_measures_chatgpt_reference_and_run_variation() -> None:
    reference = {
        ("paper-1", "novelty"): 5.0,
        ("paper-2", "novelty"): 3.0,
    }
    records = [
        _record("paper-1", "novelty", 1, 5),
        _record("paper-2", "novelty", 1, 3),
        _record("paper-1", "novelty", 2, 4),
        _record("paper-2", "novelty", 2, 3),
        _record("paper-1", "novelty", 3, 5),
        _record("paper-2", "novelty", 3, 3),
    ]
    report = build_reliability_report(
        records=records,
        reference_by_key=reference,
        run_count=3,
        metadata={"model": "test"},
    )
    assert report["aggregate"]["complete_item_count"] == 2
    assert report["aggregate"]["chatgpt_reference_mae_of_run_mean"] == 0.166667
    assert report["aggregate"]["stability"]["exact_run_agreement_rate"] == 0.5
    assert report["per_dimension"]["novelty"]["paper_count"] == 2
    assert report["interpretation_boundary"]["human_paper_quality_measured"] is False


def test_reuses_ret11b_hybrid_as_a_matching_repeat_baseline() -> None:
    primary = [{
        **_record("paper-1", "novelty", 1, 4),
        "selected_chunk_ids": ["chunk-1"],
    }]
    baseline = {
        "records": [{
            "paper_id": "paper-1",
            "dimension": "novelty",
            "system": "hybrid",
            "selected_chunk_ids": ["chunk-1"],
            "evaluation": {"status": "scored", "score": 5},
            "error": None,
        }]
    }
    result = build_heldout_repeat_stability(
        primary_records=primary,
        baseline_report=baseline,
    )
    assert result["complete_item_count"] == 1
    assert result["selection_mismatch_count"] == 0
    assert result["stability"]["mean_within_item_range"] == 1.0


def test_versioned_eval02a_summary_matches_full_report() -> None:
    summary = json.loads(
        (ROOT / "evals/academic_quality/eval02a_score_reliability_summary.json")
        .read_text(encoding="utf-8")
    )
    report_path = ROOT / "evals/academic_quality" / summary["full_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == summary[
        "full_report_sha256"
    ]
    assert report["terminal_failure_count"] == 0
    assert report["run_summaries"][0]["scorable_count"] == 42
    assert report["aggregate"]["chatgpt_reference_mae_of_run_mean"] == 0.5
    assert report["heldout_repeat_stability"]["complete_item_count"] == 16
    assert summary["interpretation"]["human_paper_quality_measured"] is False
    assert summary["interpretation"]["heldout_used_for_retuning"] is False
