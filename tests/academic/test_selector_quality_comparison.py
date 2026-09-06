from __future__ import annotations

from evals.academic_quality.compare_selector_quality import (
    _pearson,
    _spearman,
    build_report,
)
from evals.academic_quality.validate_dataset import QUALITY_DIR
import hashlib
import json


def _record(system: str, dimension: str, score: float) -> dict:
    return {
        "paper_id": "paper-1",
        "dimension": dimension,
        "system": system,
        "evaluation": {"status": "scored", "score": score},
        "error": None,
    }


def test_correlation_helpers_handle_ties_and_constant_values() -> None:
    assert _pearson([1, 2, 3], [2, 4, 6]) == 1.0
    assert _pearson([1, 1], [2, 3]) is None
    assert _spearman([1, 2, 2, 4], [1, 3, 3, 5]) == 1.0


def test_report_compares_mae_without_human_claims() -> None:
    dimensions = [
        "research_question_clarity",
        "methodological_rigor",
        "experimental_design",
        "baseline_quality",
        "result_validity",
        "reproducibility",
        "novelty",
        "limitations_discussion",
        "citation_quality",
    ]
    reference = {
        ("paper-1", dimension): float((index % 5) + 1)
        for index, dimension in enumerate(dimensions)
    }
    records = []
    for dimension in dimensions:
        target = reference[("paper-1", dimension)]
        records.append(_record("section", dimension, target))
        records.append(_record("hybrid", dimension, max(1.0, target - 1.0)))

    report = build_report(
        records=records,
        reference_by_key=reference,
        input_metadata={"model": "test-model"},
    )

    assert report["summary"]["section"]["dimension_score_mae"] == 0.0
    assert report["summary"]["section"]["failure_count"] == 0
    assert report["summary"]["section"]["insufficient_evidence_count"] == 0
    assert report["summary"]["hybrid"]["dimension_score_mae"] > 0.0
    assert report["winner_on_chatgpt_reference_mae"] == "section"
    assert report["interpretation_boundary"]["human_paper_quality_measured"] is False
    assert report["interpretation_boundary"]["retuning_from_test_permitted"] is False


def test_report_separates_insufficient_evidence_from_failures() -> None:
    dimension = "research_question_clarity"
    records = [
        {
            "paper_id": "paper-1",
            "dimension": dimension,
            "system": system,
            "evaluation": {
                "status": "insufficient_evidence",
                "score": None,
            },
            "error": None,
        }
        for system in ("section", "hybrid")
    ]

    report = build_report(
        records=records,
        reference_by_key={("paper-1", dimension): 4.0},
        input_metadata={"model": "test-model"},
    )

    for system in ("section", "hybrid"):
        assert report["summary"][system]["failure_count"] == 0
        assert report["summary"][system]["insufficient_evidence_count"] == 1
        assert report["summary"][system]["score_missing_rate"] == 1.0


def test_versioned_ret11b_summary_matches_full_report() -> None:
    summary = json.loads(
        (QUALITY_DIR / "ret11b_selector_quality_summary.json").read_text()
    )
    report_path = QUALITY_DIR / summary["full_report"]
    report = json.loads(report_path.read_text())

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == summary[
        "full_report_sha256"
    ]
    assert report["summary"]["section"]["failure_count"] == 0
    assert report["summary"]["hybrid"]["failure_count"] == 0
    assert report["summary"]["section"]["dimension_score_mae"] == summary[
        "section"
    ]["dimension_score_mae"]
    assert report["summary"]["hybrid"]["dimension_score_mae"] == summary[
        "hybrid"
    ]["dimension_score_mae"]
    assert summary["interpretation"]["human_paper_quality_measured"] is False
    assert summary["interpretation"]["retuning_permitted"] is False
