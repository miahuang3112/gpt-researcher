import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

from gpt_researcher.academic.paper_chunker import (
    AcademicPaperChunker,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect chunks generated from an academic PDF."
    )

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the academic PDF.",
    )

    parser.add_argument(
        "--paper-id",
        default=None,
        help="Paper ID. Defaults to the PDF filename.",
    )

    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional source URL or DOI URL.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    paper_id = args.paper_id or pdf_path.stem

    chunker = AcademicPaperChunker(
        target_tokens=800,
        overlap_tokens=120,
        min_chunk_tokens=100,
    )

    chunks = chunker.chunk_pdf(
        file_path=pdf_path,
        paper_id=paper_id,
        source_url=args.source_url,
    )

    token_counts = [
        chunker.count_tokens(chunk.text)
        for chunk in chunks
    ]

    page_numbers = sorted(
        {
            chunk.page
            for chunk in chunks
            if chunk.page is not None
        }
    )

    section_counts = Counter(
        chunk.section or "Unclassified"
        for chunk in chunks
    )

    print("\nPDF CHUNK SUMMARY")
    print(f"PDF: {pdf_path}")
    print(f"Paper ID: {paper_id}")
    print(f"Pages containing chunks: {len(page_numbers)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Minimum tokens: {min(token_counts)}")
    print(f"Maximum tokens: {max(token_counts)}")
    print(f"Average tokens: {mean(token_counts):.2f}")

    print("\nSECTION COUNTS")

    for section, count in section_counts.items():
        print(f"{section}: {count}")

    print("\nCHUNK PREVIEW")

    for index, (chunk, token_count) in enumerate(
        zip(chunks, token_counts),
        start=1,
    ):
        section = chunk.section or "Unclassified"
        preview = " ".join(chunk.text.split())

        if len(preview) > 160:
            preview = f"{preview[:160]}..."

        print(
            f"[{index:03d}] "
            f"page={chunk.page} "
            f"section={section!r} "
            f"tokens={token_count} "
            f"id={chunk.chunk_id}"
        )
        print(f"      {preview}")

    oversized_chunks = [
        count
        for count in token_counts
        if count > (
            chunker.target_tokens
            + chunker.min_chunk_tokens
        )
    ]

    if oversized_chunks:
        print(
            "\nWARNING: "
            f"{len(oversized_chunks)} chunks exceeded "
            "the expected soft token limit."
        )
    else:
        print("\nToken length check: PASS")


if __name__ == "__main__":
    main()