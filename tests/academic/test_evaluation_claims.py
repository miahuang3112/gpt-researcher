import pytest

from gpt_researcher.academic.evaluation_claims import (
    build_evaluation_claims,
)
from gpt_researcher.academic.schemas import DimensionEvaluation


def make_evaluation() -> DimensionEvaluation:
    return DimensionEvaluation(
        dimension="methodological_rigor",
        status="scored",
        score=4,
        rationale="The experimental design is appropriate.",
        strengths=[
            "The baselines are clearly described.",
            " Multiple datasets are used. ",
        ],
        concerns=[
            "The sample size is limited.",
        ],
        evidence_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-1",
        ],
    )


def test_builds_claims_in_expected_order():
    claims = build_evaluation_claims(
        paper_id="paper-1",
        evaluation=make_evaluation(),
    )

    assert len(claims) == 4

    assert claims[0].text == (
        "The experimental design is appropriate."
    )
    assert claims[0].claim_type == "other"

    assert claims[1].text == "The sample size is limited."
    assert claims[1].claim_type == "limitation"

    assert claims[2].text == (
        "The baselines are clearly described."
    )
    assert claims[3].text == "Multiple datasets are used."


def test_copies_and_deduplicates_evidence_ids():
    claims = build_evaluation_claims(
        paper_id="paper-1",
        evaluation=make_evaluation(),
    )

    assert all(
        claim.evidence_chunk_ids == ["chunk-1", "chunk-2"]
        for claim in claims
    )


def test_claim_ids_are_stable():
    evaluation = make_evaluation()

    first = build_evaluation_claims(
        paper_id="paper-1",
        evaluation=evaluation,
    )
    second = build_evaluation_claims(
        paper_id="paper-1",
        evaluation=evaluation,
    )

    assert [claim.claim_id for claim in first] == [
        claim.claim_id for claim in second
    ]


def test_removes_duplicate_and_empty_statements():
    evaluation = make_evaluation()
    evaluation.strengths = [
        "",
        "   ",
        "Repeated statement.",
        " repeated   statement. ",
    ]

    claims = build_evaluation_claims(
        paper_id="paper-1",
        evaluation=evaluation,
        max_claims=10,
    )

    matching = [
        claim
        for claim in claims
        if claim.text.casefold() == "repeated statement."
    ]

    assert len(matching) == 1


def test_respects_max_claims():
    claims = build_evaluation_claims(
        paper_id="paper-1",
        evaluation=make_evaluation(),
        max_claims=2,
    )

    assert len(claims) == 2


@pytest.mark.parametrize(
    ("paper_id", "max_claims"),
    [
        ("", 4),
        ("   ", 4),
        ("paper-1", 0),
        ("paper-1", -1),
    ],
)
def test_rejects_invalid_arguments(
    paper_id: str,
    max_claims: int,
):
    with pytest.raises(ValueError):
        build_evaluation_claims(
            paper_id=paper_id,
            evaluation=make_evaluation(),
            max_claims=max_claims,
        )