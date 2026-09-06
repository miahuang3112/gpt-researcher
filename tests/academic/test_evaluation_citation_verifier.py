import pytest

from gpt_researcher.academic.evaluation_citation_verifier import (
    EvaluationCitationVerifierError,
    build_claim_support_results,
    verify_evaluation_claims,
)
from gpt_researcher.academic.schemas import (
    CitationValidation,
    DimensionEvaluation,
    EvidenceChunk,
)


class FakeCitationValidator:
    def __init__(
        self,
        *,
        status: str = "supported",
        returned_paper_id: str | None = None,
        wrong_claim_id: bool = False,
    ):
        self.status = status
        self.returned_paper_id = returned_paper_id
        self.wrong_claim_id = wrong_claim_id
        self.calls = []

    async def validate_claim(
        self,
        claim,
        evidence_chunks,
    ):
        self.calls.append(
            {
                "claim": claim,
                "evidence_chunks": list(evidence_chunks),
            }
        )

        claim_id = (
            "wrong-claim-id"
            if self.wrong_claim_id
            else claim.claim_id
        )

        return CitationValidation(
            citation_id=f"citation-{claim_id}",
            claim_id=claim_id,
            paper_id=self.returned_paper_id or claim.paper_id,
            evidence_chunk_ids=[
                chunk.chunk_id
                for chunk in evidence_chunks
            ],
            status=self.status,
            explanation="Fake validation result.",
            support_score=1.0 if self.status == "supported" else 0.5,
        )


def make_evaluation() -> DimensionEvaluation:
    return DimensionEvaluation(
        dimension="methodological_rigor",
        status="scored",
        score=4,
        rationale="The experimental design is appropriate.",
        strengths=[
            "The baselines are clearly described.",
        ],
        concerns=[
            "The sample size is limited.",
        ],
        evidence_chunk_ids=[
            "chunk-1",
            "chunk-2",
        ],
    )


def make_evidence() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id="chunk-1",
            paper_id="paper-1",
            text="Evidence from the methods section.",
            section="Methods",
        ),
        EvidenceChunk(
            chunk_id="chunk-2",
            paper_id="paper-1",
            text="Evidence from the experiments section.",
            section="Experiments",
        ),
    ]


@pytest.mark.asyncio
async def test_verifies_each_evaluation_claim():
    validator = FakeCitationValidator()

    validations = await verify_evaluation_claims(
        validator=validator,
        paper_id="paper-1",
        evaluation=make_evaluation(),
        evidence_chunks=make_evidence(),
        max_claims=3,
    )

    assert len(validations) == 3
    assert len(validator.calls) == 3

    assert all(
        validation.status == "supported"
        for validation in validations
    )


@pytest.mark.asyncio
async def test_passes_only_matching_evidence_chunks():
    evaluation = make_evaluation()
    evaluation.evidence_chunk_ids = [
        "chunk-2",
        "missing-chunk",
    ]

    validator = FakeCitationValidator()

    await verify_evaluation_claims(
        validator=validator,
        paper_id="paper-1",
        evaluation=evaluation,
        evidence_chunks=make_evidence(),
    )

    for call in validator.calls:
        assert [
            chunk.chunk_id
            for chunk in call["evidence_chunks"]
        ] == ["chunk-2"]


def test_only_supported_status_counts_as_true():
    statuses = [
        "supported",
        "partially_supported",
        "contradicted",
        "insufficient_evidence",
        "not_found",
    ]

    validations = [
        CitationValidation(
            citation_id=f"citation-{index}",
            claim_id=f"claim-{index}",
            paper_id="paper-1",
            evidence_chunk_ids=["chunk-1"],
            status=status,
            explanation="Test result.",
            support_score=1.0,
        )
        for index, status in enumerate(statuses)
    ]

    assert build_claim_support_results(validations) == [
        True,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.asyncio
async def test_rejects_unexpected_claim_id():
    validator = FakeCitationValidator(
        wrong_claim_id=True,
    )

    with pytest.raises(
        EvaluationCitationVerifierError,
        match="unexpected claim_id",
    ):
        await verify_evaluation_claims(
            validator=validator,
            paper_id="paper-1",
            evaluation=make_evaluation(),
            evidence_chunks=make_evidence(),
        )


@pytest.mark.asyncio
async def test_rejects_unexpected_paper_id():
    validator = FakeCitationValidator(
        returned_paper_id="another-paper",
    )

    with pytest.raises(
        EvaluationCitationVerifierError,
        match="unexpected paper_id",
    ):
        await verify_evaluation_claims(
            validator=validator,
            paper_id="paper-1",
            evaluation=make_evaluation(),
            evidence_chunks=make_evidence(),
        )