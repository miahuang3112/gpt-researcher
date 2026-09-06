import pytest

from gpt_researcher.academic.evidence_metrics import (
    calculate_evidence_metrics,
)
from gpt_researcher.academic.schemas import EvidenceChunk
from gpt_researcher.academic.verification_policy import (
    determine_verification_status,
)


def make_chunk(
    chunk_id: str,
    section: str,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        text="Substantial academic evidence. " * 20,
        section=section,
    )


def make_complete_chunks() -> list[EvidenceChunk]:
    return [
        make_chunk("chunk-1", "Methods"),
        make_chunk("chunk-2", "Methods"),
        make_chunk("chunk-3", "Experiments"),
        make_chunk("chunk-4", "Experiments"),
    ]


def test_checks_pass_when_all_requirements_are_met() -> None:
    chunks = make_complete_chunks()

    metrics = calculate_evidence_metrics(
        chunks,
        [
            "chunk-1",
            "chunk-2",
            "chunk-3",
            "chunk-4",
        ],
        claim_support_results=[
            True,
            True,
            True,
            True,
        ],
    )

    result = determine_verification_status(metrics)

    assert result.status == "checks_passed"
    assert result.reasons == [
        "All configured automated evidence checks passed."
    ]


def test_no_selected_chunks_is_insufficient() -> None:
    metrics = calculate_evidence_metrics([], [])

    result = determine_verification_status(metrics)

    assert result.status == "insufficient_evidence"
    assert any(
        "No evidence chunks" in reason
        for reason in result.reasons
    )


def test_too_few_citations_is_insufficient() -> None:
    chunks = make_complete_chunks()

    metrics = calculate_evidence_metrics(
        chunks,
        ["chunk-1"],
        claim_support_results=[True],
    )

    result = determine_verification_status(metrics)

    assert result.status == "insufficient_evidence"
    assert any(
        "Too few valid evidence chunks" in reason
        for reason in result.reasons
    )


def test_missing_support_evaluation_requires_review() -> None:
    chunks = make_complete_chunks()

    metrics = calculate_evidence_metrics(
        chunks,
        [
            "chunk-1",
            "chunk-2",
            "chunk-3",
            "chunk-4",
        ],
    )

    result = determine_verification_status(metrics)

    assert result.status == "review_required"
    assert any(
        "has not been independently evaluated" in reason
        for reason in result.reasons
    )


def test_quality_problems_require_review() -> None:
    chunks = make_complete_chunks()

    metrics = calculate_evidence_metrics(
        chunks,
        [
            "chunk-1",
            "chunk-1",
            "chunk-2",
            "outside-prompt",
        ],
        claim_support_results=[
            True,
            False,
        ],
        unresolved_contradiction_count=1,
    )

    result = determine_verification_status(metrics)

    assert result.status == "review_required"

    combined_reasons = " ".join(result.reasons)

    assert "invalid citation" in combined_reasons
    assert "duplicate citation" in combined_reasons
    assert "Citation coverage" in combined_reasons
    assert "enough sections" in combined_reasons
    assert "Claim support rate" in combined_reasons
    assert "unresolved contradiction" in combined_reasons


def test_thresholds_are_configurable() -> None:
    chunks = make_complete_chunks()

    metrics = calculate_evidence_metrics(
        chunks,
        ["chunk-1", "chunk-3"],
        claim_support_results=[True, True],
    )

    result = determine_verification_status(
        metrics,
        minimum_citation_coverage=0.50,
        minimum_cited_sections=2,
    )

    assert result.status == "checks_passed"


def test_rejects_invalid_policy_configuration() -> None:
    metrics = calculate_evidence_metrics([], [])

    with pytest.raises(
        ValueError,
        match="minimum_cited_chunks",
    ):
        determine_verification_status(
            metrics,
            minimum_cited_chunks=0,
        )

    with pytest.raises(
        ValueError,
        match="minimum_citation_coverage",
    ):
        determine_verification_status(
            metrics,
            minimum_citation_coverage=1.1,
        )

    with pytest.raises(
        ValueError,
        match="minimum_claim_support_rate",
    ):
        determine_verification_status(
            metrics,
            minimum_claim_support_rate=-0.1,
        )

    with pytest.raises(
        ValueError,
        match="minimum_cited_sections",
    ):
        determine_verification_status(
            metrics,
            minimum_cited_sections=0,
        )