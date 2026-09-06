from pathlib import Path

import fitz
import pytest

from gpt_researcher.academic.paper_chunker import (
    AcademicPaperChunker,
    PaperPage,
)


def make_chunker() -> AcademicPaperChunker:
    return AcademicPaperChunker(
        target_tokens=60,
        overlap_tokens=10,
        min_chunk_tokens=10,
    )


def test_preserves_page_number_and_section_across_pages():
    chunker = make_chunker()

    pages = [
        PaperPage(
            page_number=1,
            text=(
                "1 Introduction\n"
                "Retrieval augmented generation combines retrieval "
                "with language model generation."
            ),
        ),
        PaperPage(
            page_number=2,
            text=(
                "The retrieved evidence provides external information "
                "for the language model."
            ),
        ),
    ]

    chunks = chunker.chunk_pages(
        pages=pages,
        paper_id="paper-001",
        source_url="https://example.com/paper.pdf",
    )

    assert len(chunks) == 2

    assert chunks[0].page == 1
    assert chunks[0].section == "Introduction"

    assert chunks[1].page == 2
    assert chunks[1].section == "Introduction"

    assert all(chunk.paper_id == "paper-001" for chunk in chunks)
    assert all(
        chunk.source_url == "https://example.com/paper.pdf"
        for chunk in chunks
    )


def test_detects_multiple_sections_on_one_page():
    chunker = make_chunker()

    pages = [
        PaperPage(
            page_number=1,
            text=(
                "1 Introduction\n"
                "This paper studies academic retrieval.\n\n"
                "2 Methods\n"
                "We compare BM25 with vector retrieval.\n\n"
                "3 Results\n"
                "Hybrid retrieval obtains higher recall."
            ),
        )
    ]

    chunks = chunker.chunk_pages(
        pages=pages,
        paper_id="paper-002",
    )

    sections = [chunk.section for chunk in chunks]

    assert sections == [
        "Introduction",
        "Methods",
        "Results",
    ]


def test_long_text_is_split_into_multiple_chunks():
    chunker = AcademicPaperChunker(
        target_tokens=40,
        overlap_tokens=8,
        min_chunk_tokens=5,
    )

    long_text = (
        "Evidence grounding improves retrieval quality and citation accuracy. "
        * 80
    )

    chunks = chunker.chunk_pages(
        pages=[
            PaperPage(
                page_number=1,
                text=f"1 Introduction\n{long_text}",
            )
        ],
        paper_id="paper-003",
    )

    assert len(chunks) > 1

    for chunk in chunks:
        token_count = chunker.count_tokens(chunk.text)

        # 短尾块合并后最多允许超出min_chunk_tokens。
        assert token_count <= (
            chunker.target_tokens
            + chunker.min_chunk_tokens
        )

        assert chunk.page == 1
        assert chunk.section == "Introduction"


def test_short_section_is_not_deleted():
    chunker = AcademicPaperChunker()

    chunks = chunker.chunk_pages(
        pages=[
            PaperPage(
                page_number=1,
                text="Abstract\nA short but important result.",
            )
        ],
        paper_id="paper-004",
    )

    assert len(chunks) == 1
    assert chunks[0].section == "Abstract"
    assert "important result" in chunks[0].text


def test_chunk_ids_are_stable():
    chunker = make_chunker()

    pages = [
        PaperPage(
            page_number=1,
            text=(
                "Methods\n"
                "We evaluate BM25 and vector retrieval."
            ),
        )
    ]

    first_result = chunker.chunk_pages(
        pages=pages,
        paper_id="paper-005",
    )

    second_result = chunker.chunk_pages(
        pages=pages,
        paper_id="paper-005",
    )

    first_ids = [chunk.chunk_id for chunk in first_result]
    second_ids = [chunk.chunk_id for chunk in second_result]

    assert first_ids == second_ids


def test_empty_pages_raise_clear_error():
    chunker = make_chunker()

    with pytest.raises(
        ValueError,
        match="no extractable text",
    ):
        chunker.chunk_pages(
            pages=[
                PaperPage(page_number=1, text=""),
                PaperPage(page_number=2, text="   "),
            ],
            paper_id="paper-006",
        )


@pytest.mark.parametrize(
    ("target_tokens", "overlap_tokens", "min_chunk_tokens"),
    [
        (0, 0, 1),
        (100, 100, 10),
        (100, 101, 10),
        (100, 10, 101),
    ],
)
def test_invalid_chunk_parameters_raise_error(
    target_tokens,
    overlap_tokens,
    min_chunk_tokens,
):
    with pytest.raises(ValueError):
        AcademicPaperChunker(
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            min_chunk_tokens=min_chunk_tokens,
        )


def test_chunk_pdf_reads_an_actual_pdf(tmp_path: Path):
    pdf_path = tmp_path / "sample_paper.pdf"

    document = fitz.open()
    page = document.new_page()

    page.insert_text(
        (72, 72),
        (
            "1 Introduction\n"
            "This PDF tests page-aware academic chunking.\n\n"
            "2 Methods\n"
            "The system uses BM25 and vector retrieval."
        ),
    )

    document.save(str(pdf_path))
    document.close()

    chunker = make_chunker()

    chunks = chunker.chunk_pdf(
        file_path=pdf_path,
        paper_id="paper-pdf",
    )

    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[0].section == "Introduction"
    assert chunks[1].section == "Methods"
    assert all(chunk.paper_id == "paper-pdf" for chunk in chunks)

def test_detects_appendix_after_references():
    chunker = make_chunker()

    pages = [
        PaperPage(
            page_number=1,
            text=(
                "References\n"
                "Smith et al. Retrieval methods."
            ),
        ),
        PaperPage(
            page_number=2,
            text=(
                "A MBR DECODING: ADDITIONAL RESULTS\n"
                "This appendix presents additional results."
            ),
        ),
        PaperPage(
            page_number=3,
            text=(
                "A.2 COMPARISON WITH UNIVERSAL SELF-CONSISTENCY\n"
                "We compare the two decoding methods."
            ),
        ),
    ]

    chunks = chunker.chunk_pages(
        pages=pages,
        paper_id="appendix-paper",
    )

    sections_by_page = {
        chunk.page: chunk.section
        for chunk in chunks
    }

    assert sections_by_page[1] == "References"
    assert sections_by_page[2] == "Appendix"
    assert sections_by_page[3] == "Appendix"


def test_removes_repeated_page_headers():
    chunker = make_chunker()

    repeated_header = (
        "Published as a conference paper at ICLR 2025"
    )

    pages = [
        PaperPage(
            page_number=1,
            text=(
                f"{repeated_header}\n"
                "1 Introduction\n"
                "This paper studies retrieval."
            ),
        ),
        PaperPage(
            page_number=2,
            text=(
                f"{repeated_header}\n"
                "2 Methods\n"
                "We use hybrid retrieval."
            ),
        ),
        PaperPage(
            page_number=3,
            text=(
                f"{repeated_header}\n"
                "3 Results\n"
                "Hybrid retrieval improves recall."
            ),
        ),
    ]

    chunks = chunker.chunk_pages(
        pages=pages,
        paper_id="header-paper",
    )

    assert all(
        repeated_header not in chunk.text
        for chunk in chunks
    )

    assert {
        chunk.section
        for chunk in chunks
    } == {
        "Introduction",
        "Methods",
        "Results",
    }