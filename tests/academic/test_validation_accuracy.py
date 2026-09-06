import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.academic_quality.measure_validation_accuracy import (
    ALLOWED_RELATIONS,
    CITATION_STATUSES,
    QUALITY_DIR,
    binary_metrics,
    build_report,
    classification_metrics,
    run_evaluation,
    validate_eval02b_dataset,
)
from gpt_researcher.academic.contradiction_policy import (
    FROZEN_CONTRADICTION_POLICY,
)


ROOT = Path(__file__).resolve().parents[2]
CITATION_LABELS = QUALITY_DIR / "citation_validation_labels.chatgpt.test.v1.jsonl"
CONTRADICTION_LABELS = QUALITY_DIR / "contradiction_labels.chatgpt.test.v1.jsonl"
CORPUS = (
    ROOT
    / "evals/academic_retrieval/datasets/ret09-five-paper-v1/test/corpus.jsonl"
)
PROVENANCE = QUALITY_DIR / "eval02b_provenance.chatgpt.test.v1.json"


def test_eval02b_dataset_is_balanced_and_heldout_only() -> None:
    citations, contradictions, chunks = validate_eval02b_dataset(
        CITATION_LABELS, CONTRADICTION_LABELS, CORPUS
    )

    assert len(citations) == 16
    assert len(contradictions) == 12
    assert {item["gold_status"] for item in citations} == CITATION_STATUSES
    assert {item["gold_relation"] for item in contradictions} == ALLOWED_RELATIONS
    assert all(item["split"] == "test" for item in [*citations, *contradictions])
    assert all(
        chunks[chunk_id].paper_id == item["paper_id"]
        for item in citations
        for chunk_id in item["claim"]["evidence_chunk_ids"]
    )


def test_eval02b_provenance_matches_frozen_test_corpus() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    assert provenance["annotator_name"] == "ChatGPT"
    assert provenance["independent_human_review"] is False
    assert provenance["split"] == "test"
    assert hashlib.sha256(CORPUS.read_bytes()).hexdigest() == provenance[
        "frozen_test_corpus_sha256"
    ]
    assert set(provenance["excluded_development_paper_ids"]) == {
        "arxiv:1706.03762",
        "arxiv:1810.04805",
        "arxiv:2005.11401",
    }


def test_classification_metrics_count_failures_as_incorrect() -> None:
    labels = [
        {"id": "one", "gold": "a"},
        {"id": "two", "gold": "b"},
    ]
    result = classification_metrics(
        labels,
        {"one": {"predicted": "a"}, "two": {"error": {"code": "timeout"}}},
        id_field="id",
        gold_field="gold",
        predicted_field="predicted",
        classes=["a", "b"],
    )

    assert result["accuracy"] == 0.5
    assert result["failure_count"] == 1
    assert result["confusion_matrix"]["b"]["__error__"] == 1


def test_binary_metrics_reports_precision_recall_and_f1() -> None:
    result = binary_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert result == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_report_uses_frozen_warning_threshold_and_ai_boundary() -> None:
    citations = [{"case_id": "c", "gold_status": "supported"}]
    contradictions = [{
        "pair_id": "p",
        "gold_relation": "contradiction",
        "gold_is_warning": True,
    }]
    report = build_report(
        citation_labels=citations,
        contradiction_labels=contradictions,
        citation_predictions={"c": {"status": "supported"}},
        contradiction_predictions={
            "p": {"relation": "contradiction", "confidence": 0.95}
        },
        metadata={"model": "test"},
        completed_at="2026-01-01T00:00:00+00:00",
    )

    assert report["citation_validation"]["accuracy"] == 1.0
    assert report["contradiction_detection"]["accuracy"] == 1.0
    assert report["contradiction_detection"]["warning_confidence_threshold"] == (
        FROZEN_CONTRADICTION_POLICY.warning_confidence_threshold
    )
    assert report["interpretation_boundary"]["independent_human_review"] is False
    assert report["interpretation_boundary"]["heldout_used_for_retuning"] is False


class GoldCitationEvaluator:
    def __init__(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses

    async def validate_claim(self, claim, chunks):
        return SimpleNamespace(
            status=self.statuses[claim.claim_id],
            support_score=1.0,
            explanation="fixture",
        )


class GoldContradictionEvaluator:
    def __init__(self, relations: dict[str, str]) -> None:
        self.relations = relations

    async def compare_claims(self, claim_a, claim_b):
        return SimpleNamespace(
            relation=self.relations[claim_a.claim_id],
            confidence=0.99,
            condition_differences=[],
            explanation="fixture",
        )


@pytest.mark.asyncio
async def test_runner_checkpoints_and_scores_all_frozen_cases(tmp_path: Path) -> None:
    citations = [json.loads(line) for line in CITATION_LABELS.read_text().splitlines()]
    contradictions = [
        json.loads(line) for line in CONTRADICTION_LABELS.read_text().splitlines()
    ]
    args = argparse.Namespace(
        citation_labels=str(CITATION_LABELS),
        contradiction_labels=str(CONTRADICTION_LABELS),
        corpus=str(CORPUS),
        provenance=str(PROVENANCE),
        checkpoint=str(tmp_path / "checkpoint.json"),
        output=str(tmp_path / "report.json"),
        model="fixture-model",
        base_url="http://unused",
        max_attempts=1,
        retry_failures=False,
    )
    report = await run_evaluation(
        args,
        citation_evaluator=GoldCitationEvaluator({
            item["claim"]["claim_id"]: item["gold_status"] for item in citations
        }),
        contradiction_evaluator=GoldContradictionEvaluator({
            item["claim_a"]["claim_id"]: item["gold_relation"]
            for item in contradictions
        }),
    )

    assert report["citation_validation"]["accuracy"] == 1.0
    assert report["contradiction_detection"]["accuracy"] == 1.0
    assert len(json.loads((tmp_path / "checkpoint.json").read_text())[
        "citation_predictions"
    ]) == 16
    assert (tmp_path / "report.json").exists()


def test_versioned_eval02b_summary_matches_full_report() -> None:
    summary = json.loads(
        (QUALITY_DIR / "eval02b_validation_accuracy_summary.json").read_text(
            encoding="utf-8"
        )
    )
    report_path = QUALITY_DIR / summary["full_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == summary[
        "full_report_sha256"
    ]
    assert report["citation_validation"]["failure_count"] == 0
    assert report["citation_validation"]["accuracy"] == 0.8125
    assert report["contradiction_detection"]["failure_count"] == 0
    assert report["contradiction_detection"]["accuracy"] == 0.833333
    assert report["contradiction_detection"]["warning_confidence_threshold"] == 0.95
    assert summary["interpretation"]["heldout_used_for_retuning"] is False
