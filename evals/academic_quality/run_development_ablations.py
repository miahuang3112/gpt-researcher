from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from evals.academic_quality.compare_selector_quality import _pearson, _spearman
from evals.academic_quality.measure_score_reliability import _safe_error
from evals.academic_retrieval.dataset import (
    load_corpus_jsonl,
    load_queries_jsonl,
)
from evals.academic_retrieval.metrics import recall_at_k
from gpt_researcher.academic.ollama_paper_evaluator_client import (
    OllamaPaperEvaluatorClient,
)
from gpt_researcher.academic.paper_evaluation_pipeline import (
    PaperEvaluationPipeline,
)
from gpt_researcher.academic.schemas import DimensionEvaluation, EvidenceChunk


QUALITY_DIR = Path(__file__).resolve().parent
CONFIG_IDS = {
    "hybrid_anchored",
    "hybrid_no_anchors",
    "section_anchored",
}
SCORING_ANCHORS = """Scoring anchors:
1 = serious flaws or the requirement is largely unmet
2 = substantial weaknesses
3 = adequate but with meaningful limitations
4 = strong with only limited weaknesses
5 = exceptional and comprehensively supported

"""


class EvaluationClient(Protocol):
    model: str

    async def generate(self, prompt: str) -> str: ...


class _UnusedEvaluationClient:
    async def generate(self, prompt: str) -> str:
        raise AssertionError("Section selection must not call the scoring model.")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def remove_scoring_anchors(prompt: str) -> str:
    if SCORING_ANCHORS not in prompt:
        raise ValueError("Production prompt no longer contains the frozen anchor block.")
    return prompt.replace(SCORING_ANCHORS, "", 1)


def load_development_inputs(
    *,
    protocol_path: Path,
    labels_path: Path,
    corpus_path: Path,
    queries_path: Path,
    retrieval_report_path: Path,
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, list[EvidenceChunk]],
    dict[tuple[str, str], list[str]],
    dict[tuple[str, str], list[str]],
    dict[str, Any],
]:
    protocol = _load_json(protocol_path)
    if protocol.get("status") != "frozen_before_model_run":
        raise ValueError("EVAL-03 protocol was not frozen before inference.")
    if protocol.get("selection_scope") != "development-only":
        raise ValueError("EVAL-03 protocol is not development-only.")
    paper_ids = set(protocol["paper_ids"])
    if paper_ids & set(protocol["excluded_heldout_paper_ids"]):
        raise ValueError("EVAL-03 development and held-out papers overlap.")
    expected_hashes = protocol["input_hashes"]
    observed_hashes = {
        "quality_labels_sha256": _sha256(labels_path),
        "development_corpus_sha256": _sha256(corpus_path),
        "development_queries_sha256": _sha256(queries_path),
        "ret09_development_report_sha256": _sha256(retrieval_report_path),
    }
    for name, observed in observed_hashes.items():
        if expected_hashes.get(name) != observed:
            raise ValueError(f"Frozen EVAL-03 input hash changed: {name}.")

    reference: dict[tuple[str, str], float] = {}
    for paper in _load_jsonl(labels_path):
        if paper["paper_id"] not in paper_ids:
            continue
        for dimension in paper["dimensions"]:
            reference[(paper["paper_id"], dimension["dimension"])] = float(
                dimension["score"]
            )
    if len(reference) != len(paper_ids) * 9:
        raise ValueError("Development reference labels do not cover 3 x 9 dimensions.")

    chunks_by_paper: dict[str, list[EvidenceChunk]] = defaultdict(list)
    for chunk in load_corpus_jsonl(corpus_path):
        if chunk.paper_id not in paper_ids:
            raise ValueError("Development corpus contains a non-development paper.")
        chunks_by_paper[chunk.paper_id].append(chunk)
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunks in chunks_by_paper.values()
        for chunk in chunks
    }

    retrieval = _load_json(retrieval_report_path)
    selected = retrieval.get("selected_config", {})
    trials = [
        trial
        for trial in retrieval.get("trials", [])
        if trial["config"]["candidate_top_k"] == selected.get("candidate_top_k")
        and trial["config"]["rrf_k"] == selected.get("rrf_k")
    ]
    if len(trials) != 1 or retrieval.get("test_split_accessed") is not False:
        raise ValueError("RET-09 development selection is missing or accessed test data.")
    hybrid: dict[tuple[str, str], list[str]] = {}
    for query in trials[0]["queries"]:
        key = (query["paper_id"], query["dimension"])
        hybrid[key] = query["systems"]["hybrid"]["retrieved_chunk_ids"][:4]
    if set(hybrid) != set(reference):
        raise ValueError("Frozen Hybrid selections do not cover development labels.")
    if any(chunk_id not in chunks_by_id for ids in hybrid.values() for chunk_id in ids):
        raise ValueError("Frozen Hybrid selection references a missing chunk.")

    section_selector = PaperEvaluationPipeline(
        client=_UnusedEvaluationClient(), max_chunks_per_dimension=4
    )
    section = {
        key: [chunk.chunk_id for chunk in section_selector.select_evidence(
            key[1], chunks_by_paper[key[0]]
        )]
        for key in reference
    }
    queries = {query.query_id: query for query in load_queries_jsonl(queries_path)}
    retrieval_metrics: dict[str, Any] = {}
    for name, selections in {"hybrid": hybrid, "section": section}.items():
        recalls = []
        for paper_id, dimension in sorted(reference):
            query = queries[f"{paper_id}-{dimension}"]
            relevant = {
                chunk_id
                for chunk_id, grade in query.relevance_by_chunk_id.items()
                if grade > 0
            }
            recalls.append(recall_at_k(selections[(paper_id, dimension)], relevant, 4))
        retrieval_metrics[name] = {
            "query_count": len(recalls),
            "mean_recall_at_4": round(sum(recalls) / len(recalls), 6),
        }
    return reference, chunks_by_paper, hybrid, section, retrieval_metrics


def summarize_config(
    records: Sequence[dict[str, Any]],
    reference: dict[tuple[str, str], float],
    config_id: str,
) -> dict[str, Any]:
    selected = [record for record in records if record["config_id"] == config_id]
    scores: dict[tuple[str, str], float] = {}
    insufficient = 0
    terminal = 0
    for record in selected:
        if record.get("error"):
            terminal += 1
            continue
        evaluation = record.get("evaluation") or {}
        if evaluation.get("status") != "scored" or evaluation.get("score") is None:
            insufficient += 1
            continue
        scores[(record["paper_id"], record["dimension"])] = float(
            evaluation["score"]
        )
    keys = sorted(scores)
    predicted = [scores[key] for key in keys]
    targets = [reference[key] for key in keys]
    errors = [abs(left - right) for left, right in zip(predicted, targets, strict=True)]
    per_dimension: dict[str, Any] = {}
    for dimension in sorted({key[1] for key in reference}):
        dimension_keys = [key for key in keys if key[1] == dimension]
        dimension_errors = [abs(scores[key] - reference[key]) for key in dimension_keys]
        per_dimension[dimension] = {
            "scorable_count": len(dimension_keys),
            "mae": round(sum(dimension_errors) / len(dimension_errors), 6)
            if dimension_errors
            else None,
        }
    return {
        "expected_count": len(reference),
        "scorable_count": len(scores),
        "insufficient_evidence_count": insufficient,
        "terminal_failure_count": terminal,
        "chatgpt_reference_mae": round(sum(errors) / len(errors), 6)
        if errors
        else None,
        "chatgpt_reference_pearson": round(_pearson(predicted, targets), 6)
        if _pearson(predicted, targets) is not None
        else None,
        "chatgpt_reference_spearman": round(_spearman(predicted, targets), 6)
        if _spearman(predicted, targets) is not None
        else None,
        "exact_match_rate": round(sum(error == 0 for error in errors) / len(errors), 6)
        if errors
        else None,
        "within_one_point_rate": round(
            sum(error <= 1 for error in errors) / len(errors), 6
        )
        if errors
        else None,
        "per_dimension": per_dimension,
    }


def challenger_is_eligible(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
    *,
    minimum_mae_improvement: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if challenger["terminal_failure_count"] != 0:
        reasons.append("terminal failures are non-zero")
    if challenger["scorable_count"] < baseline["scorable_count"]:
        reasons.append("scorable count is below baseline")
    if challenger["within_one_point_rate"] < baseline["within_one_point_rate"]:
        reasons.append("within-one-point rate is below baseline")
    improvement = (
        baseline["chatgpt_reference_mae"] - challenger["chatgpt_reference_mae"]
    )
    if improvement < minimum_mae_improvement:
        reasons.append(
            f"MAE improvement {improvement:.6f} is below {minimum_mae_improvement:.6f}"
        )
    return not reasons, reasons


def policy_ablation(report: dict[str, Any], thresholds: Sequence[float]) -> dict[str, Any]:
    by_threshold = {
        round(float(item["threshold"]), 2): item
        for item in report["candidate_metrics"]
    }
    candidates = [by_threshold[round(float(value), 2)] for value in thresholds]
    return {
        "source_split": report["split"],
        "candidate_metrics": candidates,
        "selected_threshold": report["selected_threshold"],
        "selected_metrics": report["selected_metrics"],
    }


def build_report(
    *,
    records: Sequence[dict[str, Any]],
    attempt_failures: Sequence[dict[str, Any]] = (),
    reference: dict[tuple[str, str], float],
    retrieval_metrics: dict[str, Any],
    protocol: dict[str, Any],
    policy_report: dict[str, Any],
    metadata: dict[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    summaries = {
        config_id: summarize_config(records, reference, config_id)
        for config_id in sorted(CONFIG_IDS)
    }
    baseline = summaries["hybrid_anchored"]
    minimum = float(protocol["minimum_mae_improvement_for_adoption"])
    prompt_eligible, prompt_reasons = challenger_is_eligible(
        baseline, summaries["hybrid_no_anchors"], minimum_mae_improvement=minimum
    )
    retriever_eligible, retriever_reasons = challenger_is_eligible(
        baseline, summaries["section_anchored"], minimum_mae_improvement=minimum
    )
    if (
        retrieval_metrics["section"]["mean_recall_at_4"]
        < retrieval_metrics["hybrid"]["mean_recall_at_4"]
    ):
        retriever_eligible = False
        retriever_reasons.append("section Recall@4 is below frozen Hybrid Recall@4")
    return {
        "format_version": 1,
        "created_at": completed_at,
        "evaluation": "eval03-development-ablations",
        "selection_scope": "development-only",
        "inputs": metadata,
        "protocol": protocol,
        "config_summaries": summaries,
        "retrieval_metrics": retrieval_metrics,
        "policy_ablation": policy_ablation(
            policy_report, protocol["policy_ablation"]["candidate_thresholds"]
        ),
        "calibration_decisions": {
            "prompt": {
                "baseline": "hybrid_anchored",
                "challenger": "hybrid_no_anchors",
                "challenger_eligible": prompt_eligible,
                "selected": "hybrid_no_anchors" if prompt_eligible else "hybrid_anchored",
                "rejection_reasons": prompt_reasons,
            },
            "retriever": {
                "baseline": "hybrid_anchored",
                "challenger": "section_anchored",
                "challenger_eligible": retriever_eligible,
                "selected": "section_anchored" if retriever_eligible else "hybrid_anchored",
                "rejection_reasons": retriever_reasons,
            },
            "contradiction_warning_threshold": {
                "selected": policy_report["selected_threshold"],
                "policy_changed": False,
                "reason": "DATA-05B already applied the frozen development-only rule.",
            },
        },
        "records": list(records),
        "attempt_failures": list(attempt_failures),
        "recovery_protocol": {
            "applies_only_to_previously_terminal_items": True,
            "same_model_temperature_dimension_and_chunk_ids": True,
            "max_chars_per_chunk": 1600,
            "compact_response": True,
        },
        "interpretation_boundary": {
            "reference_annotator_type": "ai",
            "reference_annotator_name": "ChatGPT",
            "heldout_used": False,
            "inferential_significance_claimed": False,
            "caveat": (
                "Three development papers and one AI reference annotator support "
                "configuration calibration only, not generalisation claims."
            ),
        },
    }


async def run_ablations(
    args: argparse.Namespace,
    *,
    client: EvaluationClient | None = None,
) -> dict[str, Any]:
    protocol_path = Path(args.protocol)
    labels_path = Path(args.labels)
    corpus_path = Path(args.corpus)
    queries_path = Path(args.queries)
    retrieval_report_path = Path(args.retrieval_report)
    policy_report_path = Path(args.policy_report)
    protocol = _load_json(protocol_path)
    reference, chunks_by_paper, hybrid, section, retrieval_metrics = (
        load_development_inputs(
            protocol_path=protocol_path,
            labels_path=labels_path,
            corpus_path=corpus_path,
            queries_path=queries_path,
            retrieval_report_path=retrieval_report_path,
        )
    )
    input_hashes = {
        "protocol_sha256": _sha256(protocol_path),
        "labels_sha256": _sha256(labels_path),
        "corpus_sha256": _sha256(corpus_path),
        "queries_sha256": _sha256(queries_path),
        "retrieval_report_sha256": _sha256(retrieval_report_path),
        "policy_report_sha256": _sha256(policy_report_path),
    }
    evaluator = client or OllamaPaperEvaluatorClient(
        model=args.model,
        base_url=args.base_url,
        timeout=600.0,
        temperature=0.0,
        think=False,
    )
    checkpoint_path = Path(args.checkpoint)
    checkpoint: dict[str, Any] = {
        "model": evaluator.model,
        "input_hashes": input_hashes,
        "records": [],
        "attempt_failures": [],
    }
    if checkpoint_path.exists():
        checkpoint = _load_json(checkpoint_path)
        if checkpoint.get("model") != evaluator.model:
            raise ValueError("Checkpoint model differs from requested model.")
        if checkpoint.get("input_hashes") != input_hashes:
            raise ValueError("Checkpoint inputs differ from frozen EVAL-03 inputs.")
    records: list[dict[str, Any]] = checkpoint["records"]
    attempt_failures: list[dict[str, Any]] = checkpoint.get(
        "attempt_failures", []
    )
    retry_keys: set[tuple[str, str, str]] = set()
    if args.retry_failures:
        terminal_records = [record for record in records if record.get("error")]
        retry_keys = {
            (record["config_id"], record["paper_id"], record["dimension"])
            for record in terminal_records
        }
        for record in terminal_records:
            attempt_failures.append({
                "config_id": record["config_id"],
                "paper_id": record["paper_id"],
                "dimension": record["dimension"],
                "error": record["error"],
                "superseded_for_retry": True,
            })
        records = [record for record in records if not record.get("error")]
        checkpoint["records"] = records
        checkpoint["attempt_failures"] = attempt_failures
    complete = {
        (record["config_id"], record["paper_id"], record["dimension"])
        for record in records
    }
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunks in chunks_by_paper.values()
        for chunk in chunks
    }
    configurations = {
        "hybrid_anchored": (hybrid, True),
        "hybrid_no_anchors": (hybrid, False),
        "section_anchored": (section, True),
    }
    for config_id in sorted(CONFIG_IDS):
        selections, anchored = configurations[config_id]
        for paper_id, dimension in sorted(reference):
            key = (config_id, paper_id, dimension)
            if key in complete:
                continue
            selected_ids = selections[(paper_id, dimension)]
            selected_chunks = [chunks_by_id[item] for item in selected_ids]
            prompt = PaperEvaluationPipeline.build_prompt(
                paper_id,
                dimension,
                selected_chunks,
                max_chars_per_chunk=1600 if key in retry_keys else None,
                compact_response=key in retry_keys,
            )
            if not anchored:
                prompt = remove_scoring_anchors(prompt)
            evaluation: DimensionEvaluation | None = None
            last_error: Exception | None = None
            for attempt in range(1, args.max_attempts + 1):
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
                        "config_id": config_id,
                        "paper_id": paper_id,
                        "dimension": dimension,
                        "attempt": attempt,
                        "error": _safe_error(exc),
                        "recovery_mode": key in retry_keys,
                    })
            records.append({
                "config_id": config_id,
                "paper_id": paper_id,
                "dimension": dimension,
                "selected_chunk_ids": selected_ids,
                "reference_score": reference[(paper_id, dimension)],
                "evaluation": evaluation.model_dump(mode="json")
                if evaluation is not None
                else None,
                "error": None
                if evaluation is not None
                else _safe_error(last_error or RuntimeError("unknown")),
                "recovery_mode": key in retry_keys,
            })
            checkpoint["records"] = records
            checkpoint["attempt_failures"] = attempt_failures
            _write_json(checkpoint_path, checkpoint)

    completed_at = checkpoint.get("completed_at") or datetime.now(timezone.utc).isoformat()
    checkpoint["completed_at"] = completed_at
    checkpoint["attempt_failures"] = attempt_failures
    _write_json(checkpoint_path, checkpoint)
    report = build_report(
        records=records,
        attempt_failures=attempt_failures,
        reference=reference,
        retrieval_metrics=retrieval_metrics,
        protocol=protocol,
        policy_report=_load_json(policy_report_path),
        metadata={"model": evaluator.model, "temperature": 0.0, **input_hashes},
        completed_at=completed_at,
    )
    _write_json(Path(args.output), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="evals/academic_quality/eval03_development_ablation_protocol.v1.json",
    )
    parser.add_argument(
        "--labels", default="evals/academic_quality/labels.chatgpt.v1.jsonl"
    )
    parser.add_argument(
        "--corpus",
        default="evals/academic_retrieval/datasets/ret09-five-paper-v1/development/corpus.jsonl",
    )
    parser.add_argument(
        "--queries",
        default="evals/academic_retrieval/datasets/ret09-five-paper-v1/development/queries.jsonl",
    )
    parser.add_argument(
        "--retrieval-report",
        default="evals/academic_retrieval/results/ret09-development-tuning-qwen3-embedding-4b.json",
    )
    parser.add_argument(
        "--policy-report",
        default="evals/academic_quality/results/data05b-contradiction-calibration.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="evals/academic_quality/results/eval03-development-ablation-checkpoint.json",
    )
    parser.add_argument(
        "--output",
        default="evals/academic_quality/results/eval03-development-ablations.json",
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
    report = asyncio.run(run_ablations(args))
    print(json.dumps({
        "config_summaries": report["config_summaries"],
        "calibration_decisions": report["calibration_decisions"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
