from gpt_researcher.academic.source_retriever import (
    CrossrefRetriever,
    OpenAlexRetriever,
    _extract_crossref_quality_metadata,
    _extract_openalex_quality_metadata,
)


def test_normalise_doi() -> None:
    doi = OpenAlexRetriever._normalise_doi(
        "https://doi.org/10.1234/EXAMPLE"
    )

    assert doi == "10.1234/example"
    assert OpenAlexRetriever._normalise_doi(None) is None


def test_reconstruct_abstract() -> None:
    inverted_index = {
        "Retrieval": [0],
        "augmented": [1],
        "generation": [2],
        "improves": [3],
        "answers": [4],
    }

    abstract = OpenAlexRetriever._reconstruct_abstract(
        inverted_index
    )

    assert abstract == (
        "Retrieval augmented generation improves answers"
    )


def test_convert_openalex_work_to_metadata() -> None:
    work = {
        "id": "https://openalex.org/W123456789",
        "title": "Example RAG Paper",
        "publication_year": 2024,
        "doi": "https://doi.org/10.1234/EXAMPLE",
        "type": "article",
        "authorships": [
            {
                "author": {
                    "display_name": "Alice Example",
                }
            },
            {
                "author": {
                    "display_name": "Bob Example",
                }
            },
        ],
        "primary_location": {
            "landing_page_url": "https://example.com/paper",
            "source": {
                "display_name": "Example Journal",
                "type": "journal",
            },
        },
        "abstract_inverted_index": {
            "This": [0],
            "paper": [1],
            "studies": [2],
            "RAG": [3],
        },
        "cited_by_count": 12,
        "open_access": {
            "is_oa": True,
        },
    }

    paper = OpenAlexRetriever._to_metadata(work)

    assert paper.paper_id == "10.1234/example"
    assert paper.title == "Example RAG Paper"
    assert paper.authors == [
        "Alice Example",
        "Bob Example",
    ]
    assert paper.year == 2024
    assert paper.doi == "10.1234/example"
    assert paper.venue == "Example Journal"
    assert paper.abstract == "This paper studies RAG"
    assert paper.citation_count == 12
    assert paper.is_open_access is True
    assert paper.source_type == "journal"
    assert paper.retrieved_from == "openalex"


def test_map_source_types() -> None:
    assert (
        OpenAlexRetriever._map_source_type(
            "preprint",
            "repository",
        )
        == "preprint"
    )

    assert (
        OpenAlexRetriever._map_source_type(
            "proceedings-article",
            "conference",
        )
        == "conference"
    )

    assert (
        OpenAlexRetriever._map_source_type(
            "article",
            "journal",
        )
        == "journal"
    )

    assert (
        OpenAlexRetriever._map_source_type(
            "book-chapter",
            None,
        )
        == "book"
    )

    assert (
        OpenAlexRetriever._map_source_type(
            "report",
            None,
        )
        == "report"
    )

def test_convert_crossref_work_to_metadata() -> None:
    item = {
        "DOI": "10.1234/CROSSREF-EXAMPLE",
        "title": [
            "Example Crossref Paper",
        ],
        "author": [
            {
                "given": "Alice",
                "family": "Example",
            },
            {
                "given": "Bob",
                "family": "Researcher",
            },
        ],
        "published-online": {
            "date-parts": [
                [2025, 6, 1],
            ],
        },
        "container-title": [
            "Example Journal",
        ],
        "type": "journal-article",
        "URL": (
            "https://doi.org/"
            "10.1234/CROSSREF-EXAMPLE"
        ),
        "abstract": (
            "<jats:p>RAG &amp; evaluation.</jats:p>"
        ),
        "is-referenced-by-count": 8,
    }

    paper = CrossrefRetriever._to_metadata(item)

    assert paper.paper_id == (
        "10.1234/crossref-example"
    )
    assert paper.title == "Example Crossref Paper"
    assert paper.authors == [
        "Alice Example",
        "Bob Researcher",
    ]
    assert paper.year == 2025
    assert paper.doi == "10.1234/crossref-example"
    assert paper.venue == "Example Journal"
    assert paper.abstract == "RAG & evaluation."
    assert paper.citation_count == 8
    assert paper.source_type == "journal"
    assert paper.retrieved_from == "crossref"


def test_crossref_extract_year() -> None:
    online_item = {
        "published-online": {
            "date-parts": [
                [2024, 3, 10],
            ],
        }
    }

    issued_item = {
        "issued": {
            "date-parts": [
                [2022],
            ],
        }
    }

    assert (
        CrossrefRetriever._extract_year(
            online_item
        )
        == 2024
    )

    assert (
        CrossrefRetriever._extract_year(
            issued_item
        )
        == 2022
    )

    assert (
        CrossrefRetriever._extract_year({})
        is None
    )


def test_crossref_source_type_mapping() -> None:
    assert (
        CrossrefRetriever._map_source_type(
            "journal-article"
        )
        == "journal"
    )

    assert (
        CrossrefRetriever._map_source_type(
            "proceedings-article"
        )
        == "conference"
    )

    assert (
        CrossrefRetriever._map_source_type(
            "posted-content"
        )
        == "preprint"
    )

    assert (
        CrossrefRetriever._map_source_type(
            "book-chapter"
        )
        == "book"
    )

    assert (
        CrossrefRetriever._map_source_type(
            "report"
        )
        == "report"
    )

    assert (
        CrossrefRetriever._map_source_type(
            "dataset"
        )
        == "unknown"
    )
def test_extract_openalex_quality_metadata() -> None:
    work = {
        "cited_by_count": 42,
        "is_retracted": False,
        "open_access": {
            "is_oa": True,
            "oa_url": "https://example.org/open-access",
        },
        "best_oa_location": {
            "pdf_url": "https://example.org/paper.pdf",
        },
    }

    result = _extract_openalex_quality_metadata(work)

    assert result == {
        "citation_count": 42,
        "is_retracted": False,
        "is_open_access": True,
        "full_text_url": "https://example.org/paper.pdf",
    }


def test_openalex_missing_quality_metadata_stays_unknown() -> None:
    result = _extract_openalex_quality_metadata({})

    assert result == {
        "citation_count": None,
        "is_retracted": None,
        "is_open_access": None,
        "full_text_url": None,
    }


def test_extract_crossref_quality_metadata() -> None:
    work = {
        "is-referenced-by-count": 17,
        "license": [
            {
                "URL": (
                    "https://creativecommons.org/"
                    "licenses/by/4.0/"
                )
            }
        ],
        "link": [
            {
                "URL": "https://example.org/article.pdf",
                "content-type": "application/pdf",
            }
        ],
    }

    result = _extract_crossref_quality_metadata(work)

    assert result == {
        "citation_count": 17,
        "is_retracted": None,
        "is_open_access": True,
        "full_text_url": "https://example.org/article.pdf",
    }


def test_crossref_unknown_status_is_not_marked_false() -> None:
    work = {
        "is-referenced-by-count": 0,
        "license": [
            {
                "URL": "https://publisher.example/license"
            }
        ],
        "link": [
            {
                "URL": "https://example.org/article",
                "content-type": "text/html",
            }
        ],
    }

    result = _extract_crossref_quality_metadata(work)

    assert result == {
        "citation_count": 0,
        "is_retracted": None,
        "is_open_access": None,
        "full_text_url": None,
    }