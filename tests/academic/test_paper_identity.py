from gpt_researcher.academic.paper_identity import (
    canonical_paper_id,
    deduplicate_papers,
    normalize_doi,
    normalize_title,
)
from gpt_researcher.academic.schemas import PaperMetadata


def paper(**values) -> PaperMetadata:
    defaults = {
        "paper_id": "source-record",
        "title": "A Retrieval-Augmented Generation Study",
        "retrieved_from": "openalex",
    }
    defaults.update(values)
    return PaperMetadata(**defaults)


def test_normalizes_doi_urls_prefixes_and_case() -> None:
    assert normalize_doi(
        " HTTPS://DX.DOI.ORG/10.1234/RAG.001?source=test "
    ) == "10.1234/rag.001"
    assert normalize_doi("doi: 10.5555/ABC;") == "10.5555/abc"
    assert normalize_doi("not-a-doi") is None


def test_normalizes_unicode_punctuation_and_whitespace_in_titles() -> None:
    assert normalize_title("ＲＡＧ:  A Study—Revisited") == (
        "rag a study revisited"
    )
    assert normalize_title("RAG: A  Study Revisited") == (
        "rag a study revisited"
    )


def test_canonical_identity_prefers_doi_then_title_hash() -> None:
    assert canonical_paper_id(
        doi="https://doi.org/10.1234/EXAMPLE",
        title="Different title",
    ) == "10.1234/example"
    assert canonical_paper_id(
        doi=None,
        title="RAG: A Study",
    ) == canonical_paper_id(
        doi=None,
        title="rag — a study",
    )


def test_deduplicates_same_doi_and_merges_best_metadata() -> None:
    result = deduplicate_papers(
        [
            paper(
                paper_id="W1",
                doi="https://doi.org/10.1234/EXAMPLE",
                authors=["Alice Example"],
                citation_count=4,
                source_ids=["W1"],
            ),
            paper(
                paper_id="crossref-1",
                doi="10.1234/example",
                authors=["Alice Example", "Bob Researcher"],
                abstract="A much more complete abstract for this paper.",
                citation_count=12,
                retrieved_from="crossref",
                source_ids=["crossref-1"],
            ),
        ]
    )

    assert len(result) == 1
    assert result[0].paper_id == "10.1234/example"
    assert result[0].doi == "10.1234/example"
    assert result[0].authors == ["Alice Example", "Bob Researcher"]
    assert result[0].citation_count == 12
    assert set(result[0].source_ids) == {"W1", "crossref-1"}


def test_attaches_doi_less_title_match_to_one_doi_record() -> None:
    result = deduplicate_papers(
        [
            paper(paper_id="W1", doi="10.1234/rag"),
            paper(
                paper_id="crossref-no-doi",
                title="A Retrieval Augmented Generation Study",
                doi=None,
            ),
        ]
    )

    assert len(result) == 1
    assert result[0].paper_id == "10.1234/rag"


def test_does_not_merge_same_title_with_conflicting_dois() -> None:
    result = deduplicate_papers(
        [
            paper(paper_id="W1", doi="10.1234/one"),
            paper(paper_id="W2", doi="10.1234/two"),
            paper(paper_id="unknown", doi=None),
        ]
    )

    assert len(result) == 3
    assert {item.paper_id for item in result} >= {
        "10.1234/one",
        "10.1234/two",
    }
