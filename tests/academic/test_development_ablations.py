import argparse
import hashlib
import json
import re
from pathlib import Path

import pytest

from evals.academic_quality.run_development_ablations import (
    QUALITY_DIR,
    build_report,
    challenger_is_eligible,
    load_development_inputs,
    remove_scoring_anchors,
    run_ablations,
    summarize_config,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = QUALITY_DIR / "eval03_development_ablation_protocol.v1.json"
LABELS = QUALITY_DIR / "labels.chatgpt.v1.jsonl"
CORPUS = ROOT / (
    "evals/academic_retrieval/datasets/ret09-five-paper-v1/"
    "development/corpus.jsonl"
)
QUERIES = CORPUS.parent / "queries.jsonl"
RETRIEVAL_REPORT = ROOT / (
    "evals/academic_retrieval/results/"
    "ret09-development-tuning-qwen3-embedding-4b.json"
)
POLICY_REPORT = QUALITY_DIR / "results/data05b-contradiction-calibration.json"
SUMMARY = QUALITY_DIR / "eval03_development_ablation_summary.json"

requires_protected_corpus = pytest.mark.skipif(
    not CORPUS.is_file() or not QUERIES.is_file(),
    reason="protected real-paper corpus is not distributed publicly",
)


def test_frozen_protocol_is_development_only() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen_before_model_run"
    assert protocol["selection_scope"] == "development-only"
    assert len(protocol["paper_ids"]) == 3
    assert set(protocol["paper_ids"]).isdisjoint(
        protocol["excluded_heldout_paper_ids"]
    )
    assert protocol["interpretation_boundary"]["heldout_access_permitted"] is False


def test_versioned_summary_pins_completed_report() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    report_path = ROOT / summary["versioned_artifacts"]["report"]

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        summary["versioned_artifacts"]["report_sha256"]
    )
    assert summary["terminal_failure_count"] == 0
    assert summary["recovered_terminal_item_count"] == 4
    assert summary["decisions"]["prompt"]["selected"] == "hybrid_anchored"
    assert summary["decisions"]["retriever"]["selected"] == "hybrid_anchored"
    assert summary["interpretation_boundary"]["heldout_used"] is False


def test_anchor_ablation_removes_only_numeric_anchor_block() -> None:
    prompt = "before\nScoring anchors:\n1 = serious flaws or the requirement is largely unmet\n2 = substantial weaknesses\n3 = adequate but with meaningful limitations\n4 = strong with only limited weaknesses\n5 = exceptional and comprehensively supported\n\nafter"

    result = remove_scoring_anchors(prompt)

    assert result == "before\nafter"


@requires_protected_corpus
def test_loads_three_paper_development_selections_and_recall() -> None:
    reference, chunks, hybrid, section, metrics = load_development_inputs(
        protocol_path=PROTOCOL,
        labels_path=LABELS,
        corpus_path=CORPUS,
        queries_path=QUERIES,
        retrieval_report_path=RETRIEVAL_REPORT,
    )

    assert len(reference) == len(hybrid) == len(section) == 27
    assert set(chunks) == {
        "arxiv:1706.03762",
        "arxiv:1810.04805",
        "arxiv:2005.11401",
    }
    assert metrics["hybrid"]["mean_recall_at_4"] == 0.222359
    assert metrics["section"]["mean_recall_at_4"] == 0.313365


def _record(config: str, paper: str, dimension: str, score: float) -> dict:
    return {
        "config_id": config,
        "paper_id": paper,
        "dimension": dimension,
        "evaluation": {"status": "scored", "score": score},
        "error": None,
    }


def test_summary_and_predeclared_adoption_threshold() -> None:
    reference = {("paper", "novelty"): 5.0}
    baseline = summarize_config(
        [_record("hybrid_anchored", "paper", "novelty", 4)],
        reference,
        "hybrid_anchored",
    )
    challenger = {**baseline, "chatgpt_reference_mae": 0.8}

    eligible, reasons = challenger_is_eligible(
        baseline, challenger, minimum_mae_improvement=0.1
    )

    assert eligible is True
    assert reasons == []
    challenger["within_one_point_rate"] = 0.0
    eligible, reasons = challenger_is_eligible(
        baseline, challenger, minimum_mae_improvement=0.1
    )
    assert eligible is False
    assert "within-one-point rate is below baseline" in reasons


class FixedEvaluationClient:
    model = "fixture-model"

    async def generate(self, prompt: str) -> str:
        dimension = re.search(
            r"Evaluation dimension:\n([^\n]+)", prompt
        ).group(1)
        chunk_id = re.search(r"chunk_id: ([^\n]+)", prompt).group(1)
        return json.dumps({
            "dimension": dimension,
            "status": "scored",
            "score": 4,
            "rationale": "Fixture result.",
            "evidence_chunk_ids": [chunk_id],
            "external_evidence_ids": [],
            "strengths": [],
            "concerns": [],
        })


class FailFirstEvaluationClient(FixedEvaluationClient):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return "not-json"
        return await super().generate(prompt)


class RecordingEvaluationClient(FixedEvaluationClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return await super().generate(prompt)


@pytest.mark.asyncio
@requires_protected_corpus
async def test_runner_scores_all_three_predeclared_configurations(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        protocol=str(PROTOCOL),
        labels=str(LABELS),
        corpus=str(CORPUS),
        queries=str(QUERIES),
        retrieval_report=str(RETRIEVAL_REPORT),
        policy_report=str(POLICY_REPORT),
        checkpoint=str(tmp_path / "checkpoint.json"),
        output=str(tmp_path / "report.json"),
        model="fixture-model",
        base_url="http://unused",
        max_attempts=1,
        retry_failures=False,
    )
    report = await run_ablations(args, client=FixedEvaluationClient())

    assert len(report["records"]) == 81
    assert {record["config_id"] for record in report["records"]} == {
        "hybrid_anchored",
        "hybrid_no_anchors",
        "section_anchored",
    }
    assert all(
        record["paper_id"] not in {"arxiv:2106.09685", "arxiv:2410.02902"}
        for record in report["records"]
    )
    assert report["interpretation_boundary"]["heldout_used"] is False
    assert report["policy_ablation"]["selected_threshold"] == 0.95
    assert (tmp_path / "checkpoint.json").exists()


@pytest.mark.asyncio
@requires_protected_corpus
async def test_retry_failures_uses_compact_audited_recovery(tmp_path: Path) -> None:
    args = argparse.Namespace(
        protocol=str(PROTOCOL),
        labels=str(LABELS),
        corpus=str(CORPUS),
        queries=str(QUERIES),
        retrieval_report=str(RETRIEVAL_REPORT),
        policy_report=str(POLICY_REPORT),
        checkpoint=str(tmp_path / "checkpoint.json"),
        output=str(tmp_path / "report.json"),
        model="fixture-model",
        base_url="http://unused",
        max_attempts=1,
        retry_failures=False,
    )
    initial = await run_ablations(args, client=FailFirstEvaluationClient())
    assert sum(bool(record.get("error")) for record in initial["records"]) == 1

    args.retry_failures = True
    recovery_client = RecordingEvaluationClient()
    recovered = await run_ablations(args, client=recovery_client)

    assert len(recovery_client.prompts) == 1
    assert "Recovery mode:" in recovery_client.prompts[0]
    assert sum(bool(record.get("error")) for record in recovered["records"]) == 0
    assert sum(bool(record.get("recovery_mode")) for record in recovered["records"]) == 1
    assert any(
        failure.get("superseded_for_retry")
        for failure in recovered["attempt_failures"]
    )
    assert recovered["recovery_protocol"]["max_chars_per_chunk"] == 1600
