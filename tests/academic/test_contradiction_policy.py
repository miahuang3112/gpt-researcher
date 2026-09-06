import hashlib
import json
from pathlib import Path

from gpt_researcher.academic.contradiction_policy import (
    FROZEN_CONTRADICTION_POLICY,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = (
    ROOT / "evals/academic_quality/data05b_frozen_contradiction_policy.json"
)


def test_frozen_contradiction_policy_matches_calibration_artifact() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy = FROZEN_CONTRADICTION_POLICY
    assert payload["status"] == "frozen"
    assert payload["selection_scope"] == "development-only"
    assert payload["policy_id"] == policy.policy_id
    assert payload["model"] == policy.model
    assert (
        payload["warning_confidence_threshold"]
        == policy.warning_confidence_threshold
    )
    assert set(payload["positive_relations"]) == policy.positive_relations


def test_frozen_policy_hashes_match_versioned_inputs() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    integrity = payload["integrity"]
    labels = ROOT / (
        "evals/academic_quality/contradiction_labels.chatgpt.dev.v1.jsonl"
    )
    report = ROOT / (
        "evals/academic_quality/results/data05b-contradiction-calibration.json"
    )
    assert hashlib.sha256(labels.read_bytes()).hexdigest() == integrity["labels_sha256"]
    assert hashlib.sha256(report.read_bytes()).hexdigest() == integrity[
        "calibration_report_sha256"
    ]


def test_frozen_policy_never_used_heldout_for_selection() -> None:
    integrity = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["integrity"]
    assert integrity["held_out_test_used_for_selection"] is False
    assert integrity["retuning_after_test_permitted"] is False
