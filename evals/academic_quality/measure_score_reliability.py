from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

from evals.academic_quality.compare_selector_quality import (
    _pearson,
    _spearman,
)
from evals.academic_quality.validate_dataset import QUALITY_DIR, validate_dataset
from evals.academic_retrieval.dataset import load_corpus_jsonl
from gpt_researcher.academic.ollama_paper_evaluator_client import (
    OllamaPaperEvaluatorClient,
)
from gpt_researcher.academic.paper_evaluation_pipeline import PaperEvaluationPipeline
from gpt_researcher.academic.retrieval_policy import FROZEN_RETRIEVAL_POLICY
from gpt_researcher.academic.schemas import DimensionEvaluation, EvidenceChunk


class EvaluationClient(Protocol):
    model: str

    async def generate(self, prompt: str) -> str: ...


class ScoreReliabilityError(ValueError):
    """EVAL-02A inputs or outputs violate the frozen measurement contract."""


def _safe_error(exc: Exception) -> dict[str, str]:
    """Classify failures without persisting model text or private details."""
    message = str(exc).lower()
    if "could not connect to ollama" in message:
        code = "ollama_unreachable"
    elif "timed out" in message or "timeout" in message:
        code = "ollama_timeout"
    elif "valid json" in message:
        code = "invalid_json"
    elif "schema" in message or "dimensionevaluation" in message:
        code = "response_schema_validation"
    elif "evidence chunks" in message:
        code = "unknown_chunk_id"
    elif "external scholarly records" in message:
        code = "unknown_external_id"
    else:
        code = "model_response_error"
    return {"type": type(exc).__name__, "code": code}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_frozen_selections(
    development_report_path: Path,
    heldout_report_path: Path,
    *,
    top_k: int = 4,
) -> tuple[dict[tuple[str, str], list[str]], dict[str, Any]]:
    development = _load_json(development_report_path)
    heldout = _load_json(heldout_report_path)
    selected = development.get("selected_config", {})
    expected = FROZEN_RETRIEVAL_POLICY
    if (
        selected.get("candidate_top_k") != expected.candidate_top_k
        or selected.get("rrf_k") != expected.rrf_k
    ):
        raise ScoreReliabilityError("Development report does not match the frozen policy.")
    trials = [
        trial
        for trial in development.get("trials", [])
        if trial["config"]["candidate_top_k"] == expected.candidate_top_k
        and trial["config"]["rrf_k"] == expected.rrf_k
    ]
    if len(trials) != 1:
        raise ScoreReliabilityError("Frozen development trial is missing or ambiguous.")
    if heldout.get("parameter_selection_permitted") is not False:
        raise ScoreReliabilityError("Held-out report does not prohibit retuning.")
    heldout_policy = heldout.get("policy", {})
    if heldout_policy.get("policy_id") != expected.policy_id:
        raise ScoreReliabilityError("Held-out report does not use the frozen policy.")

    selections: dict[tuple[str, str], list[str]] = {}
    for query in [*trials[0]["queries"], *heldout["benchmark"]["queries"]]:
        key = (query["paper_id"], query["dimension"])
        if key in selections:
            raise ScoreReliabilityError(f"Duplicate frozen query: {key}.")
        system = query["systems"]["hybrid"]
        if system.get("error"):
            raise ScoreReliabilityError(f"Frozen retrieval failed for {key}.")
        selected_ids = system["retrieved_chunk_ids"][:top_k]
        if len(selected_ids) != top_k:
            raise ScoreReliabilityError(f"Frozen retrieval is short for {key}.")
        selections[key] = selected_ids
    return selections, {
        "policy_id": expected.policy_id,
        "system": "hybrid",
        "embedding_model": expected.embedding_model,
        "candidate_top_k": expected.candidate_top_k,
        "rrf_k": expected.rrf_k,
        "evidence_top_k": top_k,
        "development_report_sha256": _sha256(development_report_path),
        "heldout_report_sha256": _sha256(heldout_report_path),
        "heldout_used_for_retuning": False,
    }


def _score_map(records: Sequence[dict[str, Any]]) -> dict[tuple[str, str, int], float]:
    result: dict[tuple[str, str, int], float] = {}
    for record in records:
        evaluation = record.get("evaluation")
        if record.get("error") or not evaluation or evaluation.get("status") != "scored":
            continue
        score = evaluation.get("score")
        if score is not None:
            result[(record["paper_id"], record["dimension"], record["run"])] = float(score)
    return result


def _stability(scores_by_run: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not scores_by_run:
        return {
            "complete_item_count": 0,
            "mean_within_item_standard_deviation": None,
            "mean_within_item_range": None,
            "exact_run_agreement_rate": None,
            "mean_pairwise_run_pearson": None,
            "mean_pairwise_run_spearman": None,
        }
    item_values = list(zip(*scores_by_run, strict=True))
    if len(scores_by_run) < 2:
        return {
            "complete_item_count": len(item_values),
            "mean_within_item_standard_deviation": None,
            "mean_within_item_range": None,
            "max_within_item_range": None,
            "exact_run_agreement_rate": None,
            "mean_pairwise_run_pearson": None,
            "mean_pairwise_run_spearman": None,
        }
    standard_deviations = [statistics.pstdev(values) for values in item_values]
    ranges = [max(values) - min(values) for values in item_values]
    pearsons = [
        value
        for left, right in combinations(scores_by_run, 2)
        if (value := _pearson(left, right)) is not None
    ]
    spearmans = [
        value
        for left, right in combinations(scores_by_run, 2)
        if (value := _spearman(left, right)) is not None
    ]
    return {
        "complete_item_count": len(item_values),
        "mean_within_item_standard_deviation": _round(_mean(standard_deviations)),
        "mean_within_item_range": _round(_mean(ranges)),
        "max_within_item_range": _round(max(ranges) if ranges else None),
        "exact_run_agreement_rate": _round(
            sum(value == 0 for value in ranges) / len(ranges) if ranges else None
        ),
        "mean_pairwise_run_pearson": _round(_mean(pearsons)),
        "mean_pairwise_run_spearman": _round(_mean(spearmans)),
    }


def build_heldout_repeat_stability(
    *,
    primary_records: Sequence[dict[str, Any]],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    baseline_records = [
        record
        for record in baseline_report.get("records", [])
        if record.get("system") == "hybrid"
    ]
    primary_by_key = {
        (record["paper_id"], record["dimension"]): record
        for record in primary_records
        if record["run"] == 1
    }
    baseline_scores: list[float] = []
    primary_scores: list[float] = []
    keys: list[tuple[str, str]] = []
    selection_mismatch_count = 0
    for baseline in baseline_records:
        key = (baseline["paper_id"], baseline["dimension"])
        primary = primary_by_key.get(key)
        if primary is None:
            continue
        if baseline.get("selected_chunk_ids") != primary.get("selected_chunk_ids"):
            selection_mismatch_count += 1
            continue
        baseline_evaluation = baseline.get("evaluation")
        primary_evaluation = primary.get("evaluation")
        if (
            baseline.get("error")
            or primary.get("error")
            or not baseline_evaluation
            or not primary_evaluation
            or baseline_evaluation.get("status") != "scored"
            or primary_evaluation.get("status") != "scored"
        ):
            continue
        baseline_scores.append(float(baseline_evaluation["score"]))
        primary_scores.append(float(primary_evaluation["score"]))
        keys.append(key)
    per_dimension: dict[str, Any] = {}
    for dimension in sorted({dimension for _, dimension in keys}):
        indices = [index for index, key in enumerate(keys) if key[1] == dimension]
        values = [
            [baseline_scores[index] for index in indices],
            [primary_scores[index] for index in indices],
        ]
        per_dimension[dimension] = _stability(values)
    return {
        "scope": "two-heldout-papers-two-independent-runs",
        "baseline": "RET-11B frozen-Hybrid run",
        "repeat": "EVAL-02A five-paper run held-out subset",
        "expected_item_count": len(baseline_records),
        "complete_item_count": len(keys),
        "selection_mismatch_count": selection_mismatch_count,
        "stability": _stability([baseline_scores, primary_scores]),
        "per_dimension": per_dimension,
        "limitation": (
            "Run-to-run stability covers only the two held-out papers and is "
            "descriptive; development papers have one scoring run."
        ),
    }


def build_reliability_report(
    *,
    records: Sequence[dict[str, Any]],
    reference_by_key: dict[tuple[str, str], float],
    run_count: int,
    metadata: dict[str, Any],
    attempt_failures: Sequence[dict[str, Any]] = (),
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    scores = _score_map(records)
    expected_keys = sorted(reference_by_key)
    run_summaries: list[dict[str, Any]] = []
    for run in range(1, run_count + 1):
        available = [key for key in expected_keys if (*key, run) in scores]
        predicted = [scores[(*key, run)] for key in available]
        reference = [reference_by_key[key] for key in available]
        errors = [abs(left - right) for left, right in zip(predicted, reference, strict=True)]
        run_summaries.append({
            "run": run,
            "scorable_count": len(available),
            "missing_count": len(expected_keys) - len(available),
            "chatgpt_reference_mae": _round(_mean(errors)),
            "chatgpt_reference_pearson": _round(_pearson(predicted, reference)),
            "chatgpt_reference_spearman": _round(_spearman(predicted, reference)),
            "exact_match_rate": _round(
                sum(error == 0 for error in errors) / len(errors) if errors else None
            ),
            "within_one_point_rate": _round(
                sum(error <= 1 for error in errors) / len(errors) if errors else None
            ),
        })

    complete_keys = [
        key
        for key in expected_keys
        if all((*key, run) in scores for run in range(1, run_count + 1))
    ]
    all_run_scores = [
        [scores[(*key, run)] for key in complete_keys]
        for run in range(1, run_count + 1)
    ]
    per_dimension: dict[str, Any] = {}
    dimensions = sorted({dimension for _, dimension in expected_keys})
    for dimension in dimensions:
        keys = [key for key in complete_keys if key[1] == dimension]
        run_scores = [
            [scores[(*key, run)] for key in keys]
            for run in range(1, run_count + 1)
        ]
        mean_predictions = [
            sum(scores[(*key, run)] for run in range(1, run_count + 1)) / run_count
            for key in keys
        ]
        references = [reference_by_key[key] for key in keys]
        per_dimension[dimension] = {
            "paper_count": len(keys),
            "chatgpt_reference_mae": _round(_mean([
                abs(prediction - reference)
                for prediction, reference in zip(mean_predictions, references, strict=True)
            ])),
            "chatgpt_reference_pearson": _round(_pearson(mean_predictions, references)),
            "chatgpt_reference_spearman": _round(_spearman(mean_predictions, references)),
            "stability": _stability(run_scores),
        }

    mean_predictions = [
        sum(scores[(*key, run)] for run in range(1, run_count + 1)) / run_count
        for key in complete_keys
    ]
    references = [reference_by_key[key] for key in complete_keys]
    aggregate_errors = [
        abs(prediction - reference)
        for prediction, reference in zip(mean_predictions, references, strict=True)
    ]
    actual_failed_calls = [
        failure
        for failure in attempt_failures
        if not failure.get("superseded_for_retry")
    ]
    superseded_terminal_records = [
        failure
        for failure in attempt_failures
        if failure.get("superseded_for_retry")
    ]
    failed_keys = {
        (failure["paper_id"], failure["dimension"], failure["run"])
        for failure in actual_failed_calls
    }
    successful_keys = {
        (record["paper_id"], record["dimension"], record["run"])
        for record in records
        if not record.get("error")
    }
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "evaluation": "eval02a-paper-score-reliability",
        "scope": "five-paper-descriptive-measurement",
        "inputs": metadata,
        "run_count": run_count,
        "expected_score_count_per_run": len(expected_keys),
        "model_call_count": (
            sum(not bool(record.get("error")) for record in records)
            + len(actual_failed_calls)
        ),
        "failed_model_call_count": len(actual_failed_calls),
        "failed_item_count": len(failed_keys),
        "recovered_item_count": len(failed_keys & successful_keys),
        "superseded_terminal_record_count": len(superseded_terminal_records),
        "terminal_failure_count": sum(bool(record.get("error")) for record in records),
        "run_summaries": run_summaries,
        "aggregate": {
            "complete_item_count": len(complete_keys),
            "chatgpt_reference_mae_of_run_mean": _round(_mean(aggregate_errors)),
            "chatgpt_reference_pearson_of_run_mean": _round(
                _pearson(mean_predictions, references)
            ),
            "chatgpt_reference_spearman_of_run_mean": _round(
                _spearman(mean_predictions, references)
            ),
            "stability": _stability(all_run_scores),
        },
        "per_dimension": per_dimension,
        "records": list(records),
        "attempt_failures": list(attempt_failures),
        "interpretation_boundary": {
            "reference_annotator_type": "ai",
            "reference_annotator_name": "ChatGPT",
            "human_paper_quality_measured": False,
            "inferential_significance_claimed": False,
            "heldout_used_for_retuning": False,
            "caveat": (
                "Five influential computer-science papers and one ChatGPT reference "
                "annotator support descriptive engineering measurement only."
            ),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def measure_score_reliability(
    *,
    development_report_path: str | Path,
    heldout_report_path: str | Path,
    development_manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    repeat_baseline_report_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    run_count: int = 1,
    model: str = "qwen3.5:9b",
    base_url: str = "http://127.0.0.1:11434",
    client: EvaluationClient | None = None,
    max_attempts: int = 3,
    concurrency: int = 2,
    retry_terminal: bool = False,
) -> dict[str, Any]:
    if run_count < 1:
        raise ScoreReliabilityError("run_count must be at least 1.")
    if max_attempts < 1:
        raise ScoreReliabilityError("max_attempts must be at least 1.")
    if concurrency < 1:
        raise ScoreReliabilityError("concurrency must be at least 1.")
    validate_dataset()
    development_report = Path(development_report_path).resolve()
    heldout_report = Path(heldout_report_path).resolve()
    development_manifest = Path(development_manifest_path).resolve()
    heldout_manifest = Path(heldout_manifest_path).resolve()
    repeat_baseline_report = Path(repeat_baseline_report_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    selections, retrieval_metadata = load_frozen_selections(
        development_report, heldout_report, top_k=FROZEN_RETRIEVAL_POLICY.evidence_top_k_default
    )

    labels_path = QUALITY_DIR / "labels.chatgpt.v1.jsonl"
    papers_path = QUALITY_DIR / "papers.v1.json"
    provenance_path = QUALITY_DIR / "provenance.chatgpt.v1.json"
    reference_by_key: dict[tuple[str, str], float] = {}
    for paper in _load_jsonl(labels_path):
        for dimension in paper["dimensions"]:
            reference_by_key[(paper["paper_id"], dimension["dimension"])] = float(
                dimension["score"]
            )
    if set(reference_by_key) != set(selections):
        raise ScoreReliabilityError("Frozen selections do not exactly cover quality labels.")

    chunks: list[EvidenceChunk] = []
    for manifest in (development_manifest, heldout_manifest):
        chunks.extend(load_corpus_jsonl(manifest.parent / "corpus.jsonl"))
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for selected_ids in selections.values():
        if any(chunk_id not in chunks_by_id for chunk_id in selected_ids):
            raise ScoreReliabilityError("A frozen selection references a missing chunk.")

    evaluator = client or OllamaPaperEvaluatorClient(
        model=model,
        base_url=base_url,
        timeout=600.0,
        temperature=0.0,
        think=False,
    )
    input_hashes = {
        "development_report_sha256": _sha256(development_report),
        "heldout_report_sha256": _sha256(heldout_report),
        "reference_labels_sha256": _sha256(labels_path),
    }
    checkpoint_payload: dict[str, Any] = {"records": []}
    if checkpoint.exists():
        checkpoint_payload = _load_json(checkpoint)
        if checkpoint_payload.get("model") != evaluator.model:
            raise ScoreReliabilityError("Checkpoint model differs from requested model.")
        if checkpoint_payload.get("input_hashes") != input_hashes:
            raise ScoreReliabilityError("Checkpoint inputs differ from current inputs.")
        if checkpoint_payload.get("run_count") != run_count:
            raise ScoreReliabilityError("Checkpoint run_count differs from requested run_count.")
    records: list[dict[str, Any]] = checkpoint_payload["records"]
    attempt_failures: list[dict[str, Any]] = checkpoint_payload.get(
        "attempt_failures", []
    )
    retry_keys: set[tuple[str, str, int]] = set()
    if retry_terminal:
        terminal_records = [record for record in records if record.get("error")]
        retry_keys = {
            (record["paper_id"], record["dimension"], record["run"])
            for record in terminal_records
        }
        for record in terminal_records:
            attempt_failures.append({
                "paper_id": record["paper_id"],
                "dimension": record["dimension"],
                "run": record["run"],
                "error": record["error"],
                "superseded_for_retry": True,
            })
        records = [record for record in records if not record.get("error")]
    completed = {
        (record["paper_id"], record["dimension"], record["run"])
        for record in records
    }
    semaphore = asyncio.Semaphore(concurrency)
    checkpoint_lock = asyncio.Lock()

    async def evaluate_one(run: int, paper_id: str, dimension: str) -> None:
        key = (paper_id, dimension, run)
        selected_ids = selections[(paper_id, dimension)]
        async with semaphore:
            selected_chunks = [chunks_by_id[chunk_id] for chunk_id in selected_ids]
            prompt = PaperEvaluationPipeline.build_prompt(
                paper_id,
                dimension,
                selected_chunks,
                max_chars_per_chunk=1600 if key in retry_keys else None,
                compact_response=key in retry_keys,
            )
            evaluation: DimensionEvaluation | None = None
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    raw = await evaluator.generate(prompt)
                    evaluation = PaperEvaluationPipeline.parse_dimension_response(
                        raw,
                        expected_dimension=dimension,
                        allowed_chunk_ids=set(selected_ids),
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    attempt_failures.append({
                        "paper_id": paper_id,
                        "dimension": dimension,
                        "run": run,
                        "attempt": attempt,
                        "error": _safe_error(exc),
                    })
        if evaluation is not None:
            record = {
                "paper_id": paper_id,
                "dimension": dimension,
                "run": run,
                "selected_chunk_ids": selected_ids,
                "reference_score": reference_by_key[(paper_id, dimension)],
                "evaluation": evaluation.model_dump(mode="json"),
                "error": None,
                "recovery_mode": key in retry_keys,
            }
        else:
            record = {
                "paper_id": paper_id,
                "dimension": dimension,
                "run": run,
                "selected_chunk_ids": selected_ids,
                "reference_score": reference_by_key[(paper_id, dimension)],
                "evaluation": None,
                "recovery_mode": key in retry_keys,
                "error": (
                    _safe_error(last_error)
                    if last_error is not None
                    else {"type": "UnknownError", "code": "unknown"}
                ),
            }
        async with checkpoint_lock:
            records.append(record)
            completed.add(key)
            checkpoint_payload = {
                "schema_version": 1,
                "model": evaluator.model,
                "run_count": run_count,
                "input_hashes": input_hashes,
                "records": records,
                "attempt_failures": attempt_failures,
            }
            _write_json(checkpoint, checkpoint_payload)

    pending = [
        evaluate_one(run, paper_id, dimension)
        for run in range(1, run_count + 1)
        for paper_id, dimension in sorted(selections)
        if (paper_id, dimension, run) not in completed
    ]
    await asyncio.gather(*pending)

    completed_at_utc = checkpoint_payload.get("completed_at_utc")
    if len(records) == len(selections) * run_count and not any(
        record.get("error") for record in records
    ):
        completed_at_utc = completed_at_utc or datetime.now(timezone.utc).isoformat()
        checkpoint_payload = {
            "schema_version": 1,
            "model": evaluator.model,
            "run_count": run_count,
            "input_hashes": input_hashes,
            "completed_at_utc": completed_at_utc,
            "records": records,
            "attempt_failures": attempt_failures,
        }
        _write_json(checkpoint, checkpoint_payload)

    report = build_reliability_report(
        records=records,
        reference_by_key=reference_by_key,
        run_count=run_count,
        attempt_failures=attempt_failures,
        generated_at_utc=completed_at_utc,
        metadata={
            "model": evaluator.model,
            "temperature": 0.0,
            "num_predict": getattr(evaluator, "num_predict", None),
            "concurrency": concurrency,
            "paper_count": len(_load_json(papers_path)["papers"]),
            "dimension_count": 9,
            "reference_labels_sha256": _sha256(labels_path),
            "reference_provenance_sha256": _sha256(provenance_path),
            "papers_sha256": _sha256(papers_path),
            "repeat_baseline_report_sha256": _sha256(repeat_baseline_report),
            "retrieval": retrieval_metadata,
            "recovery_protocol": {
                "trigger": "terminal failures only",
                "max_chars_per_chunk": 1600,
                "compact_response": True,
            },
        },
    )
    baseline_report = _load_json(repeat_baseline_report)
    if baseline_report.get("comparison") != (
        "ret11b-section-vs-frozen-hybrid-paper-quality"
    ):
        raise ScoreReliabilityError("Repeat baseline is not the RET-11B report.")
    if baseline_report.get("inputs", {}).get("model") != evaluator.model:
        raise ScoreReliabilityError("Repeat baseline model differs from EVAL-02A.")
    report["heldout_repeat_stability"] = build_heldout_repeat_stability(
        primary_records=records,
        baseline_report=baseline_report,
    )
    _write_json(output, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-report", default="evals/academic_retrieval/results/ret09-development-tuning-qwen3-embedding-4b.json")
    parser.add_argument("--heldout-report", default="evals/academic_retrieval/results/ret10b-heldout-test-qwen3-embedding-4b.json")
    parser.add_argument("--development-manifest", default="evals/academic_retrieval/datasets/ret09-five-paper-v1/development/manifest.json")
    parser.add_argument("--heldout-manifest", default="evals/academic_retrieval/datasets/ret09-five-paper-v1/test/manifest.json")
    parser.add_argument("--repeat-baseline-report", default="evals/academic_quality/results/ret11b-section-vs-hybrid-heldout.json")
    parser.add_argument("--checkpoint", default="evals/academic_quality/results/eval02a-score-reliability-checkpoint.json")
    parser.add_argument("--output", default="evals/academic_quality/results/eval02a-score-reliability.json")
    parser.add_argument("--run-count", type=int, default=1)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--retry-terminal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = asyncio.run(measure_score_reliability(
        development_report_path=args.development_report,
        heldout_report_path=args.heldout_report,
        development_manifest_path=args.development_manifest,
        heldout_manifest_path=args.heldout_manifest,
        repeat_baseline_report_path=args.repeat_baseline_report,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        run_count=args.run_count,
        model=args.model,
        base_url=args.base_url,
        max_attempts=args.max_attempts,
        concurrency=args.concurrency,
        retry_terminal=args.retry_terminal,
    ))
    print(json.dumps({
        "run_summaries": report["run_summaries"],
        "aggregate": report["aggregate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
