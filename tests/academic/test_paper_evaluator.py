import pytest

from gpt_researcher.academic.paper_evaluator import (
    DEFAULT_DIMENSION_WEIGHTS,
    REQUIRED_DIMENSIONS,
    PaperEvaluator,
    PaperEvaluatorError,
)
from gpt_researcher.academic.schemas import (
    DimensionEvaluation,
    EvidenceChunk,
)


def make_evidence_chunks() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id="chunk-methods",
            paper_id="paper-001",
            text=(
                "The proposed method combines retrieval "
                "and preference-based decoding."
            ),
            section="Methods",
            page=3,
        ),
        EvidenceChunk(
            chunk_id="chunk-results",
            paper_id="paper-001",
            text=(
                "The method improves performance over "
                "the greedy baseline."
            ),
            section="Results",
            page=7,
        ),
        EvidenceChunk(
            chunk_id="chunk-limitations",
            paper_id="paper-001",
            text=(
                "The evaluation is limited to three "
                "benchmarks and one model family."
            ),
            section="Limitations",
            page=12,
        ),
    ]


def make_complete_evaluations(
    score: float = 4.0,
) -> list[DimensionEvaluation]:
    return [
        DimensionEvaluation(
            dimension=dimension,
            status="scored",
            score=score,
            rationale=(
                f"Evidence supports the {dimension} score."
            ),
            evidence_chunk_ids=["chunk-methods"],
            strengths=[
                f"{dimension} is clearly described."
            ],
            concerns=[],
        )
        for dimension in REQUIRED_DIMENSIONS
    ]


def make_insufficient(
    evaluation: DimensionEvaluation,
) -> DimensionEvaluation:
    return DimensionEvaluation(
        dimension=evaluation.dimension,
        status="insufficient_evidence",
        score=None,
        rationale=(
            "The available chunks do not provide "
            "enough evidence for this dimension."
        ),
        evidence_chunk_ids=[],
        strengths=[],
        concerns=["More evidence is required."],
    )


def test_complete_evaluation_calculates_weighted_score():
    evaluator = PaperEvaluator()

    result = evaluator.evaluate(
        paper_id="paper-001",
        evidence_chunks=make_evidence_chunks(),
        dimension_evaluations=make_complete_evaluations(),
    )

    assert result.paper_id == "paper-001"
    assert result.overall_score == 80.0
    assert result.evidence_coverage == 1.0
    assert result.assessment == "solid"
    assert result.missing_dimensions == []

    assert [
        evaluation.dimension
        for evaluation in result.dimensions
    ] == list(REQUIRED_DIMENSIONS)


def test_unknown_evidence_id_is_rejected():
    evaluations = make_complete_evaluations()

    evaluations[0] = evaluations[0].model_copy(
        update={
            "evidence_chunk_ids": [
                "chunk-does-not-exist"
            ]
        }
    )

    with pytest.raises(
        PaperEvaluatorError,
        match="Unknown evidence IDs",
    ):
        PaperEvaluator().evaluate(
            paper_id="paper-001",
            evidence_chunks=make_evidence_chunks(),
            dimension_evaluations=evaluations,
        )


def test_scored_dimension_requires_evidence():
    evaluations = make_complete_evaluations()

    evaluations[0] = evaluations[0].model_copy(
        update={"evidence_chunk_ids": []}
    )

    with pytest.raises(
        PaperEvaluatorError,
        match="cites no evidence",
    ):
        PaperEvaluator().evaluate(
            paper_id="paper-001",
            evidence_chunks=make_evidence_chunks(),
            dimension_evaluations=evaluations,
        )


def test_duplicate_dimension_is_rejected():
    evaluations = make_complete_evaluations()

    evaluations[-1] = evaluations[0]

    with pytest.raises(
        PaperEvaluatorError,
        match="Duplicate evaluation dimension",
    ):
        PaperEvaluator().evaluate(
            paper_id="paper-001",
            evidence_chunks=make_evidence_chunks(),
            dimension_evaluations=evaluations,
        )


def test_missing_dimension_is_rejected():
    evaluations = make_complete_evaluations()[:-1]

    with pytest.raises(
        PaperEvaluatorError,
        match="Missing evaluation dimensions",
    ):
        PaperEvaluator().evaluate(
            paper_id="paper-001",
            evidence_chunks=make_evidence_chunks(),
            dimension_evaluations=evaluations,
        )


def test_low_evidence_coverage_suppresses_overall_score():
    evaluations = make_complete_evaluations()

    insufficient_dimensions = {
        "methodological_rigor",
        "experimental_design",
    }

    evaluations = [
        (
            make_insufficient(evaluation)
            if evaluation.dimension
            in insufficient_dimensions
            else evaluation
        )
        for evaluation in evaluations
    ]

    result = PaperEvaluator().evaluate(
        paper_id="paper-001",
        evidence_chunks=make_evidence_chunks(),
        dimension_evaluations=evaluations,
    )

    assert result.evidence_coverage == 0.65
    assert result.overall_score is None
    assert result.assessment == "insufficient_evidence"

    assert set(result.missing_dimensions) == {
        "methodological_rigor",
        "experimental_design",
    }


def test_small_missing_dimension_still_allows_score():
    evaluations = make_complete_evaluations()

    evaluations = [
        (
            make_insufficient(evaluation)
            if evaluation.dimension
            == "limitations_discussion"
            else evaluation
        )
        for evaluation in evaluations
    ]

    result = PaperEvaluator().evaluate(
        paper_id="paper-001",
        evidence_chunks=make_evidence_chunks(),
        dimension_evaluations=evaluations,
    )

    assert result.evidence_coverage == 0.95
    assert result.overall_score == 80.0
    assert result.assessment == "solid"

    assert result.missing_dimensions == [
        "limitations_discussion"
    ]


def test_evidence_from_another_paper_is_rejected():
    wrong_paper_evidence = [
        chunk.model_copy(
            update={"paper_id": "paper-002"}
        )
        for chunk in make_evidence_chunks()
    ]

    with pytest.raises(
        PaperEvaluatorError,
        match="belongs to paper",
    ):
        PaperEvaluator().evaluate(
            paper_id="paper-001",
            evidence_chunks=wrong_paper_evidence,
            dimension_evaluations=(
                make_complete_evaluations()
            ),
        )


def test_duplicate_evidence_chunk_id_is_rejected():
    evidence_chunks = make_evidence_chunks()

    evidence_chunks.append(evidence_chunks[0])

    with pytest.raises(
        PaperEvaluatorError,
        match="Duplicate evidence chunk ID",
    ):
        PaperEvaluator().evaluate(
            paper_id="paper-001",
            evidence_chunks=evidence_chunks,
            dimension_evaluations=(
                make_complete_evaluations()
            ),
        )


def test_invalid_dimension_weights_are_rejected():
    missing_weight = dict(
        DEFAULT_DIMENSION_WEIGHTS
    )
    missing_weight.pop("citation_quality")

    with pytest.raises(
        PaperEvaluatorError,
        match="do not match required dimensions",
    ):
        PaperEvaluator(
            dimension_weights=missing_weight
        )

    incorrect_total = dict(
        DEFAULT_DIMENSION_WEIGHTS
    )
    incorrect_total[
        "research_question_clarity"
    ] = 0.20

    with pytest.raises(
        PaperEvaluatorError,
        match="must sum to 1.0",
    ):
        PaperEvaluator(
            dimension_weights=incorrect_total
        )

    zero_weight = dict(
        DEFAULT_DIMENSION_WEIGHTS
    )
    zero_weight["citation_quality"] = 0.0

    with pytest.raises(
        PaperEvaluatorError,
        match="greater than 0",
    ):
        PaperEvaluator(
            dimension_weights=zero_weight
        )