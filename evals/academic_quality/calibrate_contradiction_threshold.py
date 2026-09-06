from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from gpt_researcher.academic.contradiction_detector import ContradictionDetector
from gpt_researcher.academic.ollama_contradiction_client import (
    OllamaContradictionClient,
)
from gpt_researcher.academic.schemas import AcademicClaim


POSITIVE_RELATIONS = {"contradiction", "partial_contradiction"}
ALLOWED_RELATIONS = POSITIVE_RELATIONS | {
    "different_conditions",
    "consistent",
    "unrelated",
    "uncertain",
}


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, int | float]:
        return self.__dict__.copy()


def load_development_labels(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Line {line_number} has no pair_id.")
        if pair_id in seen:
            raise ValueError(f"Duplicate pair_id: {pair_id}.")
        if record.get("split") != "development":
            raise ValueError(f"{pair_id} is not a development label.")
        relation = record.get("gold_relation")
        if relation not in ALLOWED_RELATIONS:
            raise ValueError(f"{pair_id} has invalid gold_relation.")
        expected_warning = relation in POSITIVE_RELATIONS
        if record.get("gold_is_warning") is not expected_warning:
            raise ValueError(f"{pair_id} has inconsistent gold_is_warning.")
        for heldout_id in {"arxiv:2106.09685", "arxiv:2410.02902"}:
            if heldout_id in record.get("source_paper_ids", []):
                raise ValueError(f"{pair_id} leaks held-out paper {heldout_id}.")
        seen.add(pair_id)
        records.append(record)
    if not records:
        raise ValueError("Contradiction calibration labels are empty.")
    return records


def calculate_binary_metrics(
    labels: Iterable[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    threshold: float,
) -> BinaryMetrics:
    tp = fp = tn = fn = 0
    for label in labels:
        prediction = predictions[label["pair_id"]]
        predicted_positive = (
            prediction["relation"] in POSITIVE_RELATIONS
            and float(prediction["confidence"]) >= threshold
        )
        gold_positive = bool(label["gold_is_warning"])
        if predicted_positive and gold_positive:
            tp += 1
        elif predicted_positive:
            fp += 1
        elif gold_positive:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        threshold=round(threshold, 2),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
    )


def select_threshold(
    labels: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    minimum_precision: float = 0.8,
) -> tuple[BinaryMetrics, list[BinaryMetrics]]:
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be between 0 and 1.")
    missing = {item["pair_id"] for item in labels}.difference(predictions)
    if missing:
        raise ValueError(f"Missing predictions: {', '.join(sorted(missing))}.")
    candidates = [
        calculate_binary_metrics(labels, predictions, value / 100)
        for value in range(0, 101)
    ]
    eligible = [item for item in candidates if item.precision >= minimum_precision]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda item: (item.f1, item.recall, item.precision, item.threshold),
    )
    return selected, candidates


def _claim(payload: dict[str, Any]) -> AcademicClaim:
    return AcademicClaim(
        claim_id=payload["claim_id"],
        paper_id=payload["paper_id"],
        text=payload["text"],
        claim_type=payload.get("claim_type", "other"),
        evidence_chunk_ids=payload.get("evidence_chunk_ids", []),
        confidence=1.0,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    labels_path = Path(args.labels)
    labels = load_development_labels(labels_path)
    checkpoint_path = Path(args.checkpoint)
    checkpoint: dict[str, Any] = {"predictions": {}}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("model", args.model) != args.model:
            raise ValueError("Checkpoint model does not match the requested model.")
        expected_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        if checkpoint.get("labels_sha256", expected_hash) != expected_hash:
            raise ValueError("Checkpoint labels do not match the current labels.")
    predictions: dict[str, dict[str, Any]] = checkpoint.setdefault("predictions", {})

    detector = ContradictionDetector(
        OllamaContradictionClient(
            model=args.model,
            base_url=args.base_url,
            temperature=0.0,
            think=False,
        )
    )
    for label in labels:
        pair_id = label["pair_id"]
        if pair_id in predictions:
            continue
        pair = await detector.compare_claims(
            _claim(label["claim_a"]),
            _claim(label["claim_b"]),
        )
        predictions[pair_id] = {
            "relation": pair.relation,
            "confidence": pair.confidence,
            "condition_differences": pair.condition_differences,
            "explanation": pair.explanation,
        }
        checkpoint.update(
            {
                "format_version": 1,
                "model": args.model,
                "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            }
        )
        _write_json(checkpoint_path, checkpoint)

    selected, candidates = select_threshold(
        labels,
        predictions,
        minimum_precision=args.minimum_precision,
    )
    relation_correct = sum(
        predictions[item["pair_id"]]["relation"] == item["gold_relation"]
        for item in labels
    )
    completed_at = checkpoint.get("completed_at")
    if not completed_at:
        completed_at = datetime.now(timezone.utc).isoformat()
        checkpoint["completed_at"] = completed_at
        _write_json(checkpoint_path, checkpoint)
    report = {
        "format_version": 1,
        "created_at": completed_at,
        "split": "development",
        "label_provenance": "ChatGPT AI labels; not independent human review",
        "labels_path": str(labels_path),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "model": args.model,
        "temperature": 0.0,
        "minimum_precision": args.minimum_precision,
        "sample_count": len(labels),
        "positive_count": sum(bool(item["gold_is_warning"]) for item in labels),
        "negative_count": sum(not bool(item["gold_is_warning"]) for item in labels),
        "selected_threshold": selected.threshold,
        "selected_metrics": selected.as_dict(),
        "relation_accuracy": round(relation_correct / len(labels), 6),
        "predictions": predictions,
        "candidate_metrics": [item.as_dict() for item in candidates],
        "limitations": [
            "Threshold and metrics use the same small development set.",
            "The labels are ChatGPT-authored controlled examples, not human annotations.",
            "Held-out contradiction accuracy remains an EVAL-02B task.",
        ],
    }
    _write_json(Path(args.output), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        default="evals/academic_quality/contradiction_labels.chatgpt.dev.v1.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        default="evals/academic_quality/results/data05b-contradiction-checkpoint.json",
    )
    parser.add_argument(
        "--output",
        default="evals/academic_quality/results/data05b-contradiction-calibration.json",
    )
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--minimum-precision", type=float, default=0.8)
    return parser


def main() -> None:
    report = asyncio.run(run_calibration(build_parser().parse_args()))
    print(json.dumps({
        "selected_threshold": report["selected_threshold"],
        "selected_metrics": report["selected_metrics"],
        "relation_accuracy": report["relation_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
