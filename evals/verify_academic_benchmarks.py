from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkVerificationError(RuntimeError):
    """Raised when a versioned benchmark artifact or boundary has drifted."""


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkVerificationError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_check(
    *,
    name: str,
    path: Path,
    expected: str,
) -> dict[str, Any]:
    observed = _sha256(path) if path.is_file() else None
    return {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "passed": observed == expected,
    }


def verify_versioned_benchmarks() -> dict[str, Any]:
    retrieval = ROOT / "evals/academic_retrieval"
    quality = ROOT / "evals/academic_quality"
    split = retrieval / "datasets/ret09-five-paper-v1"

    ret10 = _load(retrieval / "ret10b_heldout_summary.json")
    ret11a = _load(retrieval / "ret11a_selector_comparison_summary.json")
    ret11b = _load(quality / "ret11b_selector_quality_summary.json")
    eval02a = _load(quality / "eval02a_score_reliability_summary.json")
    eval02b = _load(quality / "eval02b_validation_accuracy_summary.json")
    eval03 = _load(quality / "eval03_development_ablation_summary.json")
    contradiction_policy = _load(
        quality / "data05b_frozen_contradiction_policy.json"
    )

    hash_checks = [
        _hash_check(
            name="RET-10B frozen policy",
            path=retrieval / "ret10b_frozen_policy.json",
            expected=ret10["policy_sha256"],
        ),
        _hash_check(
            name="RET-10B held-out report",
            path=retrieval / ret10["held_out_report"],
            expected=ret10["held_out_report_sha256"],
        ),
        _hash_check(
            name="RET-10B test manifest",
            path=split / "test/manifest.json",
            expected=ret10["test_manifest_sha256"],
        ),
        _hash_check(
            name="RET-11A selector report",
            path=retrieval / ret11a["full_report"],
            expected=ret11a["full_report_sha256"],
        ),
        _hash_check(
            name="RET-11B selector-quality report",
            path=quality / ret11b["full_report"],
            expected=ret11b["full_report_sha256"],
        ),
        _hash_check(
            name="EVAL-02A reliability report",
            path=quality / eval02a["full_report"],
            expected=eval02a["full_report_sha256"],
        ),
        _hash_check(
            name="EVAL-02B validation report",
            path=quality / eval02b["full_report"],
            expected=eval02b["full_report_sha256"],
        ),
        _hash_check(
            name="EVAL-03 frozen protocol",
            path=ROOT / eval03["versioned_artifacts"]["protocol"],
            expected=eval03["versioned_artifacts"]["protocol_sha256"],
        ),
        _hash_check(
            name="EVAL-03 ablation report",
            path=ROOT / eval03["versioned_artifacts"]["report"],
            expected=eval03["versioned_artifacts"]["report_sha256"],
        ),
        _hash_check(
            name="DATA-05B calibration labels",
            path=quality / "contradiction_labels.chatgpt.dev.v1.jsonl",
            expected=contradiction_policy["integrity"]["labels_sha256"],
        ),
        _hash_check(
            name="DATA-05B calibration report",
            path=quality / "results/data05b-contradiction-calibration.json",
            expected=contradiction_policy["integrity"][
                "calibration_report_sha256"
            ],
        ),
    ]

    invariants = {
        "ret10_zero_failures": ret10["failure_count"] == 0,
        "ret10_test_not_used_for_retuning": (
            ret10["interpretation"]["test_used_for_retuning"] is False
        ),
        "ret11a_zero_failures": (
            ret11a["section"]["failure_count"] == 0
            and ret11a["hybrid"]["failure_count"] == 0
            and ret11a["hybrid"]["index_failure_count"] == 0
        ),
        "ret11a_retuning_forbidden": (
            ret11a["interpretation"]["retuning_permitted"] is False
        ),
        "ret11b_zero_failures": (
            ret11b["section"]["failure_count"] == 0
            and ret11b["hybrid"]["failure_count"] == 0
        ),
        "ret11b_retuning_forbidden": (
            ret11b["interpretation"]["retuning_permitted"] is False
        ),
        "eval02a_zero_terminal_failures": eval02a["terminal_failure_count"] == 0,
        "eval02a_heldout_not_used_for_retuning": (
            eval02a["interpretation"]["heldout_used_for_retuning"] is False
        ),
        "eval02b_zero_failures": (
            eval02b["citation_validation"]["failure_count"] == 0
            and eval02b["contradiction_detection"]["failure_count"] == 0
        ),
        "eval02b_heldout_not_used_for_retuning": (
            eval02b["interpretation"]["heldout_used_for_retuning"] is False
        ),
        "eval03_zero_terminal_failures": eval03["terminal_failure_count"] == 0,
        "eval03_development_only": (
            eval03["selection_scope"] == "development-only"
            and eval03["interpretation_boundary"]["heldout_used"] is False
        ),
        "eval03_frozen_prompt_retained": (
            eval03["decisions"]["prompt"]["selected"] == "hybrid_anchored"
            and eval03["decisions"]["prompt"]["changed"] is False
        ),
        "eval03_frozen_retriever_retained": (
            eval03["decisions"]["retriever"]["selected"]
            == "hybrid_anchored"
            and eval03["decisions"]["retriever"]["changed"] is False
        ),
        "contradiction_threshold_is_frozen": (
            contradiction_policy["warning_confidence_threshold"] == 0.95
            and eval03["decisions"]["contradiction_warning_threshold"][
                "selected"
            ]
            == 0.95
        ),
    }
    failed_hashes = [check["name"] for check in hash_checks if not check["passed"]]
    failed_invariants = [name for name, passed in invariants.items() if not passed]
    report = {
        "status": "passed" if not failed_hashes and not failed_invariants else "failed",
        "hash_checks": hash_checks,
        "invariants": invariants,
        "failed_hash_checks": failed_hashes,
        "failed_invariants": failed_invariants,
    }
    if report["status"] != "passed":
        raise BenchmarkVerificationError(json.dumps(report, indent=2))
    return report


def main() -> None:
    print(json.dumps(verify_versioned_benchmarks(), indent=2))


if __name__ == "__main__":
    main()
