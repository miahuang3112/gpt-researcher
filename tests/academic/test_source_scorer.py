import pytest

from gpt_researcher.academic.schemas import PaperMetadata
from gpt_researcher.academic.source_scorer import (
    SourceScorer,
)


@pytest.fixture
def scorer() -> SourceScorer:
    return SourceScorer()


@pytest.fixture
def journal_paper() -> PaperMetadata:
    return PaperMetadata(
        paper_id="10.1234/journal-paper",
        title="A Reproducible Retrieval Study",
        authors=["Researcher A", "Researcher B"],
        year=2025,
        doi="10.1234/journal-paper",
        venue="Journal of Retrieval Research",
        url="https://example.org/paper",
        abstract="This study evaluates hybrid retrieval.",
        citation_count=120,
        is_retracted=False,
        is_open_access=True,
        full_text_url="https://example.org/paper.pdf",
        source_type="journal",
    )


def test_complete_journal_receives_high_score(
    scorer: SourceScorer,
    journal_paper: PaperMetadata,
) -> None:
    result = scorer.score(journal_paper)

    assert result.total_score == pytest.approx(96.0)
    assert result.grade == "high"
    assert result.source_type_score == 30.0
    assert result.traceability_score == 25.0
    assert result.evidence_access_score == 20.0
    assert result.metadata_completeness_score == 15.0
    assert result.citation_score == 6.0


def test_sparse_web_source_receives_low_score(
    scorer: SourceScorer,
) -> None:
    paper = PaperMetadata(
        paper_id="web-1",
        title="Company Blog Post",
        url="https://example.com/blog",
        source_type="web",
    )

    result = scorer.score(paper)

    assert result.total_score == pytest.approx(22.0)
    assert result.grade == "low"
    assert result.score_cap == 49.0


def test_complete_preprint_is_capped_at_medium(
    scorer: SourceScorer,
) -> None:
    paper = PaperMetadata(
        paper_id="arxiv:2601.12345",
        title="A New Retrieval Method",
        authors=["Researcher A"],
        year=2026,
        doi="10.48550/arXiv.2601.12345",
        venue="arXiv",
        url="https://arxiv.org/abs/2601.12345",
        abstract="A complete preprint abstract.",
        citation_count=10,
        is_retracted=False,
        is_open_access=True,
        full_text_url="https://arxiv.org/pdf/2601.12345",
        source_type="preprint",
    )

    result = scorer.score(paper)

    assert result.total_score == pytest.approx(79.0)
    assert result.grade == "medium"
    assert result.score_cap == 79.0
    assert any(
        "capped" in reason
        for reason in result.reasons
    )


def test_retracted_source_is_rejected(
    scorer: SourceScorer,
    journal_paper: PaperMetadata,
) -> None:
    retracted_paper = journal_paper.model_copy(
        update={"is_retracted": True}
    )

    result = scorer.score(retracted_paper)

    assert result.total_score == 0.0
    assert result.grade == "rejected"
    assert result.score_cap == 0.0
    assert result.reasons == [
        "Source is marked as retracted."
    ]


def test_citation_score_is_capped_at_ten(
    scorer: SourceScorer,
    journal_paper: PaperMetadata,
) -> None:
    highly_cited_paper = journal_paper.model_copy(
        update={"citation_count": 100_000}
    )

    result = scorer.score(highly_cited_paper)

    assert result.citation_score == 10.0
    assert result.total_score <= 100.0


def test_score_many_orders_sources_by_total_score(
    scorer: SourceScorer,
    journal_paper: PaperMetadata,
) -> None:
    web_source = PaperMetadata(
        paper_id="web-1",
        title="Blog Post",
        url="https://example.com/blog",
        source_type="web",
    )

    preprint = journal_paper.model_copy(
        update={
            "paper_id": "preprint-1",
            "source_type": "preprint",
        }
    )

    results = scorer.score_many(
        [
            web_source,
            preprint,
            journal_paper,
        ]
    )

    assert [
        result.paper_id
        for result in results
    ] == [
        journal_paper.paper_id,
        "preprint-1",
        "web-1",
    ]