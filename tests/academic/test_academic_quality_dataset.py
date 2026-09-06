from __future__ import annotations

import json
from pathlib import Path

from evals.academic_quality.validate_dataset import QUALITY_DIR, validate_dataset
from gpt_researcher.academic.paper_evaluator import REQUIRED_DIMENSIONS


def test_chatgpt_quality_dataset_is_complete_and_valid() -> None:
    summary = validate_dataset()

    assert summary == {
        "annotator_type": "ai",
        "development_papers": 3,
        "dimension_count": 9,
        "independent_human_review": False,
        "label_count": 45,
        "paper_count": 5,
        "test_papers": 2,
    }


def test_quality_labels_are_separate_from_retrieval_labels() -> None:
    labels_path = QUALITY_DIR / "labels.chatgpt.v1.jsonl"
    records = [json.loads(line) for line in labels_path.read_text().splitlines()]

    assert "academic_retrieval" not in str(labels_path.parent)
    assert all("query_id" not in record for record in records)
    assert all("relevance_grade" not in record for record in records)
    assert all(
        [item["dimension"] for item in record["dimensions"]]
        == list(REQUIRED_DIMENSIONS)
        for record in records
    )


def test_provenance_forbids_human_ground_truth_claim() -> None:
    provenance = json.loads(
        (QUALITY_DIR / "provenance.chatgpt.v1.json").read_text()
    )

    assert provenance["annotator_type"] == "ai"
    assert provenance["independent_human_review"] is False
    assert "human-scored benchmark" in provenance["prohibited_claims"]
    assert Path(provenance["label_file"]).name == "labels.chatgpt.v1.jsonl"
