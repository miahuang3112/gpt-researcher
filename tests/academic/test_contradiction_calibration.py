import json
from pathlib import Path

import pytest

from evals.academic_quality.calibrate_contradiction_threshold import (
    load_development_labels,
    select_threshold,
)


LABELS = Path(
    "evals/academic_quality/contradiction_labels.chatgpt.dev.v1.jsonl"
)


def test_versioned_labels_are_development_only_and_balanced() -> None:
    labels = load_development_labels(LABELS)
    assert len(labels) == 24
    assert sum(item["gold_is_warning"] for item in labels) == 8
    relations = {item["gold_relation"] for item in labels}
    assert relations == {
        "contradiction",
        "partial_contradiction",
        "different_conditions",
        "consistent",
        "unrelated",
        "uncertain",
    }


def test_label_provenance_discloses_ai_and_heldout_boundary() -> None:
    provenance = json.loads(
        Path(
            "evals/academic_quality/contradiction_provenance.chatgpt.dev.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert provenance["annotator_type"] == "ai"
    assert provenance["independent_human_review"] is False
    assert set(provenance["excluded_heldout_paper_ids"]) == {
        "arxiv:2106.09685",
        "arxiv:2410.02902",
    }


def test_selects_highest_f1_threshold_with_precision_floor() -> None:
    labels = [
        {"pair_id": "positive-high", "gold_is_warning": True},
        {"pair_id": "positive-low", "gold_is_warning": True},
        {"pair_id": "negative-mid", "gold_is_warning": False},
        {"pair_id": "negative-relation", "gold_is_warning": False},
    ]
    predictions = {
        "positive-high": {"relation": "contradiction", "confidence": 0.91},
        "positive-low": {"relation": "partial_contradiction", "confidence": 0.81},
        "negative-mid": {"relation": "contradiction", "confidence": 0.80},
        "negative-relation": {"relation": "consistent", "confidence": 0.99},
    }
    selected, _ = select_threshold(labels, predictions, minimum_precision=0.8)
    assert selected.threshold == 0.81
    assert selected.precision == 1.0
    assert selected.recall == 1.0


def test_rejects_missing_predictions() -> None:
    with pytest.raises(ValueError, match="Missing predictions"):
        select_threshold(
            [{"pair_id": "missing", "gold_is_warning": True}],
            {},
        )
