import pytest

from gpt_researcher.academic.evidence_metrics import (
    calculate_evidence_metrics,
)
from gpt_researcher.academic.schemas import EvidenceChunk


def make_chunk(
    chunk_id: str,
    section: str | None,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        text="Substantial academic evidence. " * 20,
        section=section,
    )


def test_calculates_observable_evidence_metrics() -> None:
    selected_chunks = [
        make_chunk("chunk-1", "Background"),
        make_chunk("chunk-2", "Background"),
        make_chunk("chunk-3", "Experiments"),
        make_chunk("chunk-4", "Experiments"),
    ]

    result = calculate_evidence_metrics(
        selected_chunks,
        ["chunk-1", "chunk-3", "chunk-4"],
        claim_support_results=[True, True, False],
    )

    assert result.citation_validity == 1.0
    assert result.citation_coverage == 0.75
    assert result.section_coverage == 1.0
    assert result.claim_support_rate == 0.6667

    assert result.selected_chunk_count == 4
    assert result.cited_chunk_count == 3

    assert result.selected_section_count == 2
    assert result.cited_section_count == 2

    assert result.evaluated_claim_count == 3
    assert result.supported_claim_count == 2


def test_invalid_citation_reduces_validity() -> None:
    selected_chunks = [
        make_chunk("chunk-1", "Methods"),
        make_chunk("chunk-2", "Experiments"),
    ]

    result = calculate_evidence_metrics(
        selected_chunks,
        ["chunk-1", "outside-prompt"],
    )

    assert result.citation_validity == 0.5
    assert result.citation_coverage == 0.5
    assert result.section_coverage == 0.5

    assert result.cited_chunk_count == 1
    assert result.invalid_citation_count == 1


def test_duplicate_citations_do_not_inflate_metrics() -> None:
    selected_chunks = [
        make_chunk("chunk-1", "Methods"),
        make_chunk("chunk-2", "Experiments"),
    ]

    result = calculate_evidence_metrics(
        selected_chunks,
        ["chunk-1", "chunk-1"],
    )

    assert result.cited_chunk_count == 1
    assert result.duplicate_citation_count == 1
    assert result.citation_coverage == 0.5


def test_missing_support_results_remains_unknown() -> None:
    selected_chunks = [
        make_chunk("chunk-1", "Methods"),
    ]

    result = calculate_evidence_metrics(
        selected_chunks,
        ["chunk-1"],
    )

    assert result.claim_support_rate is None
    assert result.evaluated_claim_count == 0
    assert result.supported_claim_count == 0


def test_tracks_unresolved_contradictions() -> None:
    result = calculate_evidence_metrics(
        [],
        [],
        unresolved_contradiction_count=2,
    )

    assert result.unresolved_contradiction_count == 2
    assert result.citation_validity == 0.0
    assert result.citation_coverage == 0.0


def test_rejects_duplicate_selected_chunks() -> None:
    duplicate_chunks = [
        make_chunk("chunk-1", "Methods"),
        make_chunk("chunk-1", "Experiments"),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate selected chunk ID",
    ):
        calculate_evidence_metrics(
            duplicate_chunks,
            ["chunk-1"],
        )


def test_rejects_negative_contradiction_count() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        calculate_evidence_metrics(
            [],
            [],
            unresolved_contradiction_count=-1,
        )