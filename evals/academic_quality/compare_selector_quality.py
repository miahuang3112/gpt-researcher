from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from evals.academic_quality.validate_dataset import QUALITY_DIR, validate_dataset
from evals.academic_retrieval.dataset import load_corpus_jsonl
from gpt_researcher.academic.ollama_paper_evaluator_client import (
    OllamaPaperEvaluatorClient,
)
from gpt_researcher.academic.paper_evaluation_pipeline import PaperEvaluationPipeline
from gpt_researcher.academic.paper_evaluator import DEFAULT_DIMENSION_WEIGHTS
from gpt_researcher.academic.schemas import DimensionEvaluation, EvidenceChunk


class EvaluationClient(Protocol):
    model: str

    async def generate(self, prompt: str) -> str: ...


class SelectorQualityError(ValueError):
    """RET-11B inputs or model outputs violate the frozen comparison contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum * right_sum)
    return numerator / denominator if denominator else None


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _summarize(
    records: Sequence[dict[str, Any]],
    *,
    system: str,
    reference_by_key: dict[tuple[str, str], float],
) -> dict[str, Any]:
    system_records = [record for record in records if record["system"] == system]
    generated = [record for record in system_records if record["error"] is None]
    insufficient = [
        record
        for record in generated
        if record["evaluation"]["status"] == "insufficient_evidence"
    ]
    scorable = [
        record
        for record in generated
        if record["evaluation"]["status"] == "scored"
        and record["evaluation"]["score"] is not None
    ]
    predicted = [float(record["evaluation"]["score"]) for record in scorable]
    reference = [
        reference_by_key[(record["paper_id"], record["dimension"])]
        for record in scorable
    ]
    absolute_errors = [
        abs(prediction - target)
        for prediction, target in zip(predicted, reference, strict=True)
    ]

    per_dimension: dict[str, dict[str, Any]] = {}
    for record in scorable:
        dimension = record["dimension"]
        error = abs(
            float(record["evaluation"]["score"])
            - reference_by_key[(record["paper_id"], dimension)]
        )
        entry = per_dimension.setdefault(dimension, {"absolute_errors": []})
        entry["absolute_errors"].append(error)
    dimension_summary = {
        dimension: {
            "count": len(values["absolute_errors"]),
            "mae": _round_optional(_mean(values["absolute_errors"])),
        }
        for dimension, values in per_dimension.items()
    }

    per_paper: dict[str, dict[str, Any]] = {}
    for paper_id in sorted({record["paper_id"] for record in system_records}):
        paper_records = [
            record
            for record in scorable
            if record["paper_id"] == paper_id
        ]
        paper_predicted = [float(record["evaluation"]["score"]) for record in paper_records]
        paper_reference = [
            reference_by_key[(paper_id, record["dimension"])]
            for record in paper_records
        ]
        evidence_coverage = sum(
            DEFAULT_DIMENSION_WEIGHTS[record["dimension"]]
            for record in paper_records
        )
        predicted_weighted = sum(
            float(record["evaluation"]["score"])
            * DEFAULT_DIMENSION_WEIGHTS[record["dimension"]]
            for record in paper_records
        )
        reference_weighted = sum(
            reference_by_key[(paper_id, record["dimension"])]
            * DEFAULT_DIMENSION_WEIGHTS[record["dimension"]]
            for record in paper_records
        )
        predicted_total = (
            predicted_weighted / evidence_coverage * 20
            if evidence_coverage
            else None
        )
        reference_total = (
            reference_weighted / evidence_coverage * 20
            if evidence_coverage
            else None
        )
        per_paper[paper_id] = {
            "scorable_dimension_count": len(paper_records),
            "evidence_coverage": round(evidence_coverage, 4),
            "dimension_mae": _round_optional(
                _mean(
                    [
                        abs(prediction - target)
                        for prediction, target in zip(
                            paper_predicted, paper_reference, strict=True
                        )
                    ]
                )
            ),
            "pearson": _round_optional(_pearson(paper_predicted, paper_reference)),
            "spearman": _round_optional(_spearman(paper_predicted, paper_reference)),
            "predicted_overall_score": (
                round(predicted_total, 4) if predicted_total is not None else None
            ),
            "reference_overall_score": (
                round(reference_total, 4) if reference_total is not None else None
            ),
            "overall_score_absolute_error": (
                round(abs(predicted_total - reference_total), 4)
                if predicted_total is not None and reference_total is not None
                else None
            ),
        }

    return {
        "expected_evaluation_count": len(system_records),
        "generated_evaluation_count": len(generated),
        "scorable_evaluation_count": len(scorable),
        "insufficient_evidence_count": len(insufficient),
        "failure_count": len(system_records) - len(generated),
        "score_missing_rate": (
            (len(system_records) - len(scorable)) / len(system_records)
            if system_records
            else 0.0
        ),
        "dimension_score_mae": _round_optional(_mean(absolute_errors)),
        "pearson": _round_optional(_pearson(predicted, reference)),
        "spearman": _round_optional(_spearman(predicted, reference)),
        "exact_match_rate": (
            sum(error == 0 for error in absolute_errors) / len(absolute_errors)
            if absolute_errors
            else None
        ),
        "within_one_point_rate": (
            sum(error <= 1 for error in absolute_errors) / len(absolute_errors)
            if absolute_errors
            else None
        ),
        "mean_overall_score_absolute_error": _round_optional(
            _mean(
                [
                    paper["overall_score_absolute_error"]
                    for paper in per_paper.values()
                    if paper["overall_score_absolute_error"] is not None
                ]
            )
        ),
        "per_dimension": dimension_summary,
        "per_paper": per_paper,
    }


def build_report(
    *,
    records: Sequence[dict[str, Any]],
    reference_by_key: dict[tuple[str, str], float],
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    summaries = {
        system: _summarize(
            records,
            system=system,
            reference_by_key=reference_by_key,
        )
        for system in ("section", "hybrid")
    }
    section_mae = summaries["section"]["dimension_score_mae"]
    hybrid_mae = summaries["hybrid"]["dimension_score_mae"]
    mae_delta = (
        round(hybrid_mae - section_mae, 6)
        if section_mae is not None and hybrid_mae is not None
        else None
    )
    section_overall_mae = summaries["section"][
        "mean_overall_score_absolute_error"
    ]
    hybrid_overall_mae = summaries["hybrid"][
        "mean_overall_score_absolute_error"
    ]
    overall_mae_delta = (
        round(hybrid_overall_mae - section_overall_mae, 6)
        if section_overall_mae is not None and hybrid_overall_mae is not None
        else None
    )
    winner = None
    if mae_delta is not None:
        winner = "section" if mae_delta > 0 else "hybrid" if mae_delta < 0 else "tie"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "ret11b-section-vs-frozen-hybrid-paper-quality",
        "scope": "held-out-test-descriptive-only",
        "inputs": input_metadata,
        "summary": summaries,
        "hybrid_minus_section": {
            "dimension_score_mae": mae_delta,
            "mean_overall_score_absolute_error": overall_mae_delta,
        },
        "winner_on_chatgpt_reference_mae": winner,
        "records": list(records),
        "interpretation_boundary": {
            "reference_annotator_type": "ai",
            "reference_annotator_name": "ChatGPT",
            "human_paper_quality_measured": False,
            "retuning_from_test_permitted": False,
            "inferential_significance_claimed": False,
            "caveat": "Only two held-out papers and 18 dimension labels are measured. Results describe agreement with the versioned ChatGPT reference, not human agreement or ground-truth quality.",
        },
    }


async def compare_selector_quality(
    *,
    selector_report_path: str | Path,
    test_manifest_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path,
    model: str = "qwen3.5:9b",
    base_url: str = "http://localhost:11434",
    concurrency: int = 1,
    client: EvaluationClient | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise SelectorQualityError("concurrency must be at least 1.")
    validate_dataset()
    selector_path = Path(selector_report_path).expanduser().resolve()
    manifest_path = Path(test_manifest_path).expanduser().resolve()
    resolved_output = Path(output_path).expanduser().resolve()
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    selector_report = _load_json(selector_path)
    manifest = _load_json(manifest_path)
    if selector_report.get("comparison_scope") != "held-out-test-descriptive-only":
        raise SelectorQualityError("RET-11A report is not held-out descriptive output.")
    if manifest.get("split_role") != "test" or manifest.get("tuning_permitted") is not False:
        raise SelectorQualityError("Manifest must be a frozen, non-tunable test split.")
    if selector_report["test_split"]["manifest_sha256"] != _sha256(manifest_path):
        raise SelectorQualityError("RET-11A report and test manifest hashes differ.")

    labels_path = QUALITY_DIR / "labels.chatgpt.v1.jsonl"
    provenance_path = QUALITY_DIR / "provenance.chatgpt.v1.json"
    reference_by_key: dict[tuple[str, str], float] = {}
    for paper_record in _load_jsonl(labels_path):
        if paper_record["paper_id"] not in manifest["paper_ids"]:
            continue
        for dimension in paper_record["dimensions"]:
            reference_by_key[(paper_record["paper_id"], dimension["dimension"])] = float(
                dimension["score"]
            )
    expected_keys = {
        (query["paper_id"], query["dimension"])
        for query in selector_report["queries"]
    }
    if set(reference_by_key) != expected_keys:
        raise SelectorQualityError("Reference labels do not exactly cover RET-11A test queries.")

    chunks = load_corpus_jsonl(manifest_path.parent / "corpus.jsonl")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    evaluator_client = client or OllamaPaperEvaluatorClient(
        model=model,
        base_url=base_url,
        timeout=600.0,
        temperature=0.0,
        think=False,
    )

    checkpoint_records: list[dict[str, Any]] = []
    if resolved_checkpoint.exists():
        checkpoint_document = _load_json(resolved_checkpoint)
        if checkpoint_document.get("model") != evaluator_client.model:
            raise SelectorQualityError("Checkpoint model differs from requested model.")
        if checkpoint_document.get("selector_report_sha256") != _sha256(selector_path):
            raise SelectorQualityError("Checkpoint selector-report hash differs.")
        checkpoint_records = checkpoint_document.get("records", [])
    completed_keys = {
        (record["paper_id"], record["dimension"], record["system"])
        for record in checkpoint_records
    }

    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_one(query: dict[str, Any], system: str) -> None:
        paper_id = query["paper_id"]
        dimension = query["dimension"]
        selected_ids = query["systems"][system]["selected_chunk_ids"]
        async with semaphore:
            try:
                selected_chunks: list[EvidenceChunk] = [
                    chunks_by_id[chunk_id] for chunk_id in selected_ids
                ]
                prompt = PaperEvaluationPipeline.build_prompt(
                    paper_id, dimension, selected_chunks
                )
                raw_response = await evaluator_client.generate(prompt)
                evaluation: DimensionEvaluation = (
                    PaperEvaluationPipeline.parse_dimension_response(
                        raw_response,
                        expected_dimension=dimension,
                        allowed_chunk_ids=set(selected_ids),
                    )
                )
                record = {
                    "paper_id": paper_id,
                    "dimension": dimension,
                    "system": system,
                    "selected_chunk_ids": selected_ids,
                    "reference_score": reference_by_key[(paper_id, dimension)],
                    "evaluation": evaluation.model_dump(mode="json"),
                    "error": None,
                }
            except Exception as exc:
                record = {
                    "paper_id": paper_id,
                    "dimension": dimension,
                    "system": system,
                    "selected_chunk_ids": selected_ids,
                    "reference_score": reference_by_key[(paper_id, dimension)],
                    "evaluation": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        checkpoint_records.append(record)
        completed_keys.add((paper_id, dimension, system))
        resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        resolved_checkpoint.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model": evaluator_client.model,
                    "selector_report_sha256": _sha256(selector_path),
                    "records": checkpoint_records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    pending: list[Any] = []
    for query in selector_report["queries"]:
        for system in ("section", "hybrid"):
            paper_id = query["paper_id"]
            dimension = query["dimension"]
            key = (paper_id, dimension, system)
            if key in completed_keys:
                continue
            pending.append(evaluate_one(query, system))
    await asyncio.gather(*pending)

    report = build_report(
        records=checkpoint_records,
        reference_by_key=reference_by_key,
        input_metadata={
            "model": evaluator_client.model,
            "temperature": 0.0,
            "concurrency": concurrency,
            "top_k": selector_report["config"]["top_k"],
            "paper_ids": manifest["paper_ids"],
            "paper_count": len(manifest["paper_ids"]),
            "dimension_label_count": len(reference_by_key),
            "selector_report": str(selector_path),
            "selector_report_sha256": _sha256(selector_path),
            "test_manifest": str(manifest_path),
            "test_manifest_sha256": _sha256(manifest_path),
            "reference_labels": str(labels_path),
            "reference_labels_sha256": _sha256(labels_path),
            "reference_provenance_sha256": _sha256(provenance_path),
        },
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selector effects against ChatGPT quality references."
    )
    parser.add_argument("--selector-report", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    report = await compare_selector_quality(
        selector_report_path=args.selector_report,
        test_manifest_path=args.test_manifest,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        model=args.model,
        base_url=args.base_url,
        concurrency=args.concurrency,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["hybrid_minus_section"], ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
