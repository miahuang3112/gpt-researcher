import json

import pytest

from examples.academic_paper_full_evaluation_smoke_test import (
    load_checkpoint,
    write_checkpoint,
    initialize_checkpoint,
)
from gpt_researcher.academic.schemas import (
    DimensionEvaluation,
)


def make_dimension_result() -> DimensionEvaluation:
    return DimensionEvaluation(
        dimension="methodological_rigor",
        status="scored",
        score=4.0,
        rationale=(
            "The methodology is generally rigorous."
        ),
        evidence_chunk_ids=[
            "chunk-1",
            "chunk-2",
        ],
        strengths=[
            "Clear method.",
        ],
        concerns=[],
    )


def test_checkpoint_round_trip(tmp_path):
    checkpoint_path = (
        tmp_path / "evaluation-checkpoint.json"
    )
    dimension_result = make_dimension_result()

    write_checkpoint(
        checkpoint_path,
        {
            "paper_id": "paper-1",
            "completed": False,
            "dimension_results": [
                dimension_result.model_dump(
                    mode="json"
                )
            ],
        },
    )

    restored_results = load_checkpoint(
        checkpoint_path,
        "paper-1",
    )

    assert checkpoint_path.exists()
    assert restored_results == [dimension_result]

    temporary_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.tmp"
    )
    assert not temporary_path.exists()


def test_checkpoint_rejects_different_paper_id(
    tmp_path,
):
    checkpoint_path = (
        tmp_path / "evaluation-checkpoint.json"
    )

    write_checkpoint(
        checkpoint_path,
        {
            "paper_id": "paper-1",
            "completed": False,
            "dimension_results": [],
        },
    )

    with pytest.raises(
        ValueError,
        match="paper_id does not match",
    ):
        load_checkpoint(
            checkpoint_path,
            "paper-2",
        )


def test_checkpoint_preserves_completed_status(
    tmp_path,
):
    checkpoint_path = (
        tmp_path / "evaluation-checkpoint.json"
    )
    dimension_result = make_dimension_result()

    write_checkpoint(
        checkpoint_path,
        {
            "paper_id": "paper-1",
            "completed": True,
            "dimension_results": [
                dimension_result.model_dump(
                    mode="json"
                )
            ],
            "final_evaluation": {
                "status": "completed",
            },
        },
    )

    payload = json.loads(
        checkpoint_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["completed"] is True
    assert (
        payload["final_evaluation"]["status"]
        == "completed"
    )

def test_initialize_checkpoint_before_evaluation(
    tmp_path,
):
    checkpoint_path = (
        tmp_path / "evaluation-checkpoint.json"
    )

    restored_results = initialize_checkpoint(
        checkpoint_path,
        "paper-1",
    )

    assert restored_results == []
    assert checkpoint_path.exists()

    payload = json.loads(
        checkpoint_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload == {
        "paper_id": "paper-1",
        "completed": False,
        "dimension_results": [],
    }