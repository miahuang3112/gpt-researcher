from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from evals.academic_quality.calibrate_contradiction_threshold import (
    ALLOWED_RELATIONS,
    POSITIVE_RELATIONS,
)
from evals.academic_quality.measure_score_reliability import _safe_error
from evals.academic_retrieval.dataset import load_corpus_jsonl
from gpt_researcher.academic.citation_validator import CitationValidator
from gpt_researcher.academic.contradiction_detector import ContradictionDetector
from gpt_researcher.academic.contradiction_policy import (
    FROZEN_CONTRADICTION_POLICY,
)
from gpt_researcher.academic.ollama_citation_client import OllamaCitationClient
from gpt_researcher.academic.ollama_contradiction_client import (
    OllamaContradictionClient,
)
from gpt_researcher.academic.schemas import AcademicClaim, EvidenceChunk


QUALITY_DIR = Path(__file__).resolve().parent
HELDOUT_PAPERS = {"arxiv:2106.09685", "arxiv:2410.02902"}
DEVELOPMENT_PAPERS = {
    "arxiv:1706.03762",
    "arxiv:1810.04805",
    "arxiv:2005.11401",
}
CITATION_STATUSES = {
    "supported",
    "partially_supported",
    "contradicted",
    "insufficient_evidence",
}


class CitationEvaluator(Protocol):
    async def validate_claim(
        self, claim: AcademicClaim, chunks: list[EvidenceChunk]
    ) -> Any: ...


class ContradictionEvaluator(Protocol):
    async def compare_claims(
        self, claim_a: AcademicClaim, claim_b: AcademicClaim
    ) -> Any: ...


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _claim(payload: dict[str, Any]) -> AcademicClaim:
    return AcademicClaim(
        claim_id=payload["claim_id"],
        paper_id=payload["paper_id"],
        text=payload["text"],
        claim_type=payload.get("claim_type", "other"),
        evidence_chunk_ids=payload.get("evidence_chunk_ids", []),
        confidence=1.0,
    )


def validate_eval02b_dataset(
    citation_labels_path: Path,
    contradiction_labels_path: Path,
    corpus_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, EvidenceChunk]]:
    citation_labels = _load_jsonl(citation_labels_path)
    contradiction_labels = _load_jsonl(contradiction_labels_path)
    chunks = load_corpus_jsonl(corpus_path)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    citation_ids: set[str] = set()
    citation_counts: Counter[str] = Counter()
    for label in citation_labels:
        case_id = label.get("case_id")
        if not case_id or case_id in citation_ids:
            raise ValueError(f"Missing or duplicate citation case_id: {case_id}.")
        if label.get("split") != "test":
            raise ValueError(f"{case_id} is not a test citation label.")
        if label.get("paper_id") not in HELDOUT_PAPERS:
            raise ValueError(f"{case_id} is not sourced from a held-out paper.")
        if label.get("gold_status") not in CITATION_STATUSES:
            raise ValueError(f"{case_id} has an invalid citation status.")
        claim = label.get("claim", {})
        if claim.get("paper_id") != label["paper_id"]:
            raise ValueError(f"{case_id} claim paper_id does not match its label.")
        evidence_ids = claim.get("evidence_chunk_ids", [])
        if not evidence_ids:
            raise ValueError(f"{case_id} has no evidence chunks.")
        for chunk_id in evidence_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or chunk.paper_id != label["paper_id"]:
                raise ValueError(f"{case_id} has invalid evidence chunk {chunk_id}.")
        citation_ids.add(case_id)
        citation_counts[label["gold_status"]] += 1
    if len(citation_labels) != 16 or set(citation_counts) != CITATION_STATUSES:
        raise ValueError("Citation test set must contain all four statuses and 16 cases.")
    if len(set(citation_counts.values())) != 1:
        raise ValueError("Citation test status classes must be balanced.")

    pair_ids: set[str] = set()
    relation_counts: Counter[str] = Counter()
    for label in contradiction_labels:
        pair_id = label.get("pair_id")
        if not pair_id or pair_id in pair_ids:
            raise ValueError(f"Missing or duplicate contradiction pair_id: {pair_id}.")
        if label.get("split") != "test":
            raise ValueError(f"{pair_id} is not a test contradiction label.")
        source_ids = set(label.get("source_paper_ids", []))
        if not source_ids or not source_ids.issubset(HELDOUT_PAPERS):
            raise ValueError(f"{pair_id} leaks a non-held-out source paper.")
        if source_ids & DEVELOPMENT_PAPERS:
            raise ValueError(f"{pair_id} leaks a development paper.")
        relation = label.get("gold_relation")
        if relation not in ALLOWED_RELATIONS:
            raise ValueError(f"{pair_id} has an invalid relation.")
        if label.get("gold_is_warning") is not (relation in POSITIVE_RELATIONS):
            raise ValueError(f"{pair_id} has an inconsistent warning label.")
        claim_a = label.get("claim_a", {})
        claim_b = label.get("claim_b", {})
        if claim_a.get("paper_id") == claim_b.get("paper_id"):
            raise ValueError(f"{pair_id} does not contain a cross-paper pair.")
        pair_ids.add(pair_id)
        relation_counts[relation] += 1
    if len(contradiction_labels) != 12 or set(relation_counts) != ALLOWED_RELATIONS:
        raise ValueError("Contradiction test set must contain all six relations and 12 pairs.")
    if len(set(relation_counts.values())) != 1:
        raise ValueError("Contradiction test relation classes must be balanced.")
    return citation_labels, contradiction_labels, chunks_by_id


def classification_metrics(
    labels: Sequence[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    id_field: str,
    gold_field: str,
    predicted_field: str,
    classes: Sequence[str],
) -> dict[str, Any]:
    confusion = {
        gold: {predicted: 0 for predicted in [*classes, "__error__"]}
        for gold in classes
    }
    correct = 0
    completed = 0
    for label in labels:
        gold = label[gold_field]
        prediction = predictions.get(label[id_field], {})
        predicted = prediction.get(predicted_field, "__error__")
        if predicted not in classes:
            predicted = "__error__"
        confusion[gold][predicted] += 1
        if predicted != "__error__":
            completed += 1
        if predicted == gold:
            correct += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for category in classes:
        tp = confusion[category][category]
        fp = sum(confusion[gold][category] for gold in classes if gold != category)
        fn = sum(
            count
            for predicted, count in confusion[category].items()
            if predicted != category
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[category] = {
            "support": sum(confusion[category].values()),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "sample_count": len(labels),
        "completed_count": completed,
        "failure_count": len(labels) - completed,
        "accuracy": round(correct / len(labels), 6) if labels else None,
        "macro_precision": round(
            sum(item["precision"] for item in per_class.values()) / len(classes), 6
        ),
        "macro_recall": round(
            sum(item["recall"] for item in per_class.values()) / len(classes), 6
        ),
        "macro_f1": round(
            sum(item["f1"] for item in per_class.values()) / len(classes), 6
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def binary_metrics(gold: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    tp = sum(left and right for left, right in zip(gold, predicted, strict=True))
    fp = sum(not left and right for left, right in zip(gold, predicted, strict=True))
    tn = sum(not left and not right for left, right in zip(gold, predicted, strict=True))
    fn = sum(left and not right for left, right in zip(gold, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": round((tp + tn) / len(gold), 6) if gold else None,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def build_report(
    *,
    citation_labels: Sequence[dict[str, Any]],
    contradiction_labels: Sequence[dict[str, Any]],
    citation_predictions: dict[str, dict[str, Any]],
    contradiction_predictions: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    citation_classes = sorted(CITATION_STATUSES)
    relation_classes = sorted(ALLOWED_RELATIONS)
    citation = classification_metrics(
        citation_labels,
        citation_predictions,
        id_field="case_id",
        gold_field="gold_status",
        predicted_field="status",
        classes=citation_classes,
    )
    citation["production_supported_decision"] = binary_metrics(
        [label["gold_status"] == "supported" for label in citation_labels],
        [
            citation_predictions.get(label["case_id"], {}).get("status") == "supported"
            for label in citation_labels
        ],
    )
    contradiction = classification_metrics(
        contradiction_labels,
        contradiction_predictions,
        id_field="pair_id",
        gold_field="gold_relation",
        predicted_field="relation",
        classes=relation_classes,
    )
    threshold = FROZEN_CONTRADICTION_POLICY.warning_confidence_threshold
    contradiction["production_warning_decision"] = binary_metrics(
        [bool(label["gold_is_warning"]) for label in contradiction_labels],
        [
            (
                prediction.get("relation") in POSITIVE_RELATIONS
                and float(prediction.get("confidence", -1.0)) >= threshold
            )
            for label in contradiction_labels
            for prediction in [contradiction_predictions.get(label["pair_id"], {})]
        ],
    )
    contradiction["warning_confidence_threshold"] = threshold
    return {
        "format_version": 1,
        "created_at": completed_at,
        "evaluation": "eval02b-validation-and-contradiction-accuracy",
        "split": "held-out-test",
        "inputs": metadata,
        "citation_validation": citation,
        "contradiction_detection": contradiction,
        "predictions": {
            "citation_validation": citation_predictions,
            "contradiction_detection": contradiction_predictions,
        },
        "interpretation_boundary": {
            "reference_annotator_type": "ai",
            "reference_annotator_name": "ChatGPT",
            "independent_human_review": False,
            "heldout_used_for_retuning": False,
            "inferential_significance_claimed": False,
            "caveat": (
                "Small balanced controlled test sets from two computer-science "
                "papers measure descriptive agreement with ChatGPT labels, not "
                "human agreement or real-world generalisation."
            ),
        },
    }


async def run_evaluation(
    args: argparse.Namespace,
    *,
    citation_evaluator: CitationEvaluator | None = None,
    contradiction_evaluator: ContradictionEvaluator | None = None,
) -> dict[str, Any]:
    citation_path = Path(args.citation_labels)
    contradiction_path = Path(args.contradiction_labels)
    corpus_path = Path(args.corpus)
    provenance_path = Path(args.provenance)
    citation_labels, contradiction_labels, chunks_by_id = validate_eval02b_dataset(
        citation_path, contradiction_path, corpus_path
    )
    hashes = {
        "citation_labels_sha256": _sha256(citation_path),
        "contradiction_labels_sha256": _sha256(contradiction_path),
        "corpus_sha256": _sha256(corpus_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint: dict[str, Any] = {
        "model": args.model,
        "input_hashes": hashes,
        "citation_predictions": {},
        "contradiction_predictions": {},
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("model") != args.model:
            raise ValueError("Checkpoint model differs from the requested model.")
        if checkpoint.get("input_hashes") != hashes:
            raise ValueError("Checkpoint inputs differ from the frozen test inputs.")
    citation_predictions = checkpoint.setdefault("citation_predictions", {})
    contradiction_predictions = checkpoint.setdefault(
        "contradiction_predictions", {}
    )
    if args.retry_failures:
        citation_predictions = {
            key: value for key, value in citation_predictions.items() if not value.get("error")
        }
        contradiction_predictions = {
            key: value
            for key, value in contradiction_predictions.items()
            if not value.get("error")
        }
        checkpoint["citation_predictions"] = citation_predictions
        checkpoint["contradiction_predictions"] = contradiction_predictions

    citation_evaluator = citation_evaluator or CitationValidator(
        OllamaCitationClient(
            model=args.model,
            base_url=args.base_url,
            timeout=600.0,
            temperature=0.0,
            think=False,
        )
    )
    contradiction_evaluator = contradiction_evaluator or ContradictionDetector(
        OllamaContradictionClient(
            model=args.model,
            base_url=args.base_url,
            timeout=600.0,
            temperature=0.0,
            think=False,
        )
    )

    for label in citation_labels:
        case_id = label["case_id"]
        if case_id in citation_predictions:
            continue
        claim = _claim(label["claim"])
        chunks = [chunks_by_id[item] for item in claim.evidence_chunk_ids]
        try:
            result = await citation_evaluator.validate_claim(claim, chunks)
            citation_predictions[case_id] = {
                "status": result.status,
                "support_score": result.support_score,
                "explanation": result.explanation,
                "error": None,
            }
        except Exception as exc:
            citation_predictions[case_id] = {"error": _safe_error(exc)}
        _write_json(checkpoint_path, checkpoint)

    for label in contradiction_labels:
        pair_id = label["pair_id"]
        if pair_id in contradiction_predictions:
            continue
        last_error: Exception | None = None
        for _ in range(args.max_attempts):
            try:
                result = await contradiction_evaluator.compare_claims(
                    _claim(label["claim_a"]), _claim(label["claim_b"])
                )
                contradiction_predictions[pair_id] = {
                    "relation": result.relation,
                    "confidence": result.confidence,
                    "condition_differences": result.condition_differences,
                    "explanation": result.explanation,
                    "error": None,
                }
                break
            except Exception as exc:
                last_error = exc
        else:
            contradiction_predictions[pair_id] = {
                "error": _safe_error(last_error or RuntimeError("unknown"))
            }
        _write_json(checkpoint_path, checkpoint)

    completed_at = checkpoint.get("completed_at") or datetime.now(timezone.utc).isoformat()
    checkpoint["completed_at"] = completed_at
    _write_json(checkpoint_path, checkpoint)
    report = build_report(
        citation_labels=citation_labels,
        contradiction_labels=contradiction_labels,
        citation_predictions=citation_predictions,
        contradiction_predictions=contradiction_predictions,
        completed_at=completed_at,
        metadata={
            "model": args.model,
            "temperature": 0.0,
            "citation_case_count": len(citation_labels),
            "contradiction_pair_count": len(contradiction_labels),
            "frozen_warning_policy_id": FROZEN_CONTRADICTION_POLICY.policy_id,
            **hashes,
        },
    )
    _write_json(Path(args.output), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--citation-labels",
        default="evals/academic_quality/citation_validation_labels.chatgpt.test.v1.jsonl",
    )
    parser.add_argument(
        "--contradiction-labels",
        default="evals/academic_quality/contradiction_labels.chatgpt.test.v1.jsonl",
    )
    parser.add_argument(
        "--corpus",
        default="evals/academic_retrieval/datasets/ret09-five-paper-v1/test/corpus.jsonl",
    )
    parser.add_argument(
        "--provenance",
        default="evals/academic_quality/eval02b_provenance.chatgpt.test.v1.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="evals/academic_quality/results/eval02b-validation-checkpoint.json",
    )
    parser.add_argument(
        "--output",
        default="evals/academic_quality/results/eval02b-validation-accuracy.json",
    )
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-failures", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_attempts < 1:
        raise ValueError("max_attempts must be at least one.")
    report = asyncio.run(run_evaluation(args))
    print(json.dumps({
        "citation_validation": report["citation_validation"],
        "contradiction_detection": report["contradiction_detection"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
