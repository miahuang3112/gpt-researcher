from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpt_researcher.academic.paper_chunker import (
    AcademicPaperChunker,
)
from gpt_researcher.academic.paper_evaluator import (
    REQUIRED_DIMENSIONS,
)
from gpt_researcher.academic.schemas import (
    EvidenceChunk,
)
from gpt_researcher.academic.retrieval_queries import (
    DIMENSION_RETRIEVAL_QUERIES,
)


ANNOTATION_FIELDS = (
    "query_id",
    "paper_id",
    "dimension",
    "query",
    "chunk_id",
    "page",
    "section",
    "relevance_grade",
    "annotator_notes",
    "text",
)


def _atomic_write_text(
    path: Path,
    content: str,
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def build_query_templates(
    paper_id: str,
) -> list[dict[str, str]]:
    normalized_paper_id = paper_id.strip()

    if not normalized_paper_id:
        raise ValueError("paper_id must not be blank")

    return [
        {
            "query_id": f"{normalized_paper_id}-{dimension}",
            "paper_id": normalized_paper_id,
            "dimension": dimension,
            "query": DIMENSION_RETRIEVAL_QUERIES[dimension],
        }
        for dimension in REQUIRED_DIMENSIONS
    ]


def render_jsonl(
    records: Sequence[dict[str, Any]],
) -> str:
    return "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )


def render_annotation_tsv(
    *,
    chunks: Sequence[EvidenceChunk],
    query_templates: Sequence[dict[str, str]],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=ANNOTATION_FIELDS,
        delimiter="\t",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()

    for query_template in query_templates:
        for chunk in chunks:
            writer.writerow(
                {
                    **query_template,
                    "chunk_id": chunk.chunk_id,
                    "page": (
                        "" if chunk.page is None else chunk.page
                    ),
                    "section": chunk.section or "",
                    "relevance_grade": "",
                    "annotator_notes": "",
                    "text": chunk.text,
                }
            )

    return output.getvalue()


def export_benchmark_artifacts(
    *,
    chunks: Sequence[EvidenceChunk],
    paper_id: str,
    output_dir: str | Path,
    source_document: str,
    source_sha256: str,
    source_url: str | None = None,
    overwrite: bool = False,
    chunking_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    chunk_list = list(chunks)

    if not chunk_list:
        raise ValueError("chunks must not be empty")

    normalized_paper_id = paper_id.strip()

    if not normalized_paper_id:
        raise ValueError("paper_id must not be blank")

    if any(
        chunk.paper_id != normalized_paper_id
        for chunk in chunk_list
    ):
        raise ValueError(
            "every chunk must belong to the exported paper_id"
        )

    chunk_ids = [chunk.chunk_id for chunk in chunk_list]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk IDs must be unique")

    query_templates = build_query_templates(
        normalized_paper_id
    )
    resolved_output_dir = Path(output_dir).expanduser().resolve()

    paths = {
        "corpus": resolved_output_dir / "corpus.jsonl",
        "query_templates": (
            resolved_output_dir / "query_templates.jsonl"
        ),
        "annotations": (
            resolved_output_dir / "annotation_tasks.tsv"
        ),
        "manifest": resolved_output_dir / "manifest.json",
    }

    corpus_records = [
        chunk.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "bm25_score",
                "vector_score",
                "fused_score",
            },
        )
        for chunk in chunk_list
    ]
    manifest = {
        "format_version": 1,
        "paper_id": normalized_paper_id,
        "source_document": source_document,
        "source_sha256": source_sha256,
        "source_url": source_url,
        "chunk_count": len(chunk_list),
        "query_count": len(query_templates),
        "annotation_task_count": (
            len(chunk_list) * len(query_templates)
        ),
        "relevance_scale": {
            "0": "irrelevant",
            "1": "partially_relevant",
            "2": "highly_relevant",
        },
        "chunking": chunking_config or {},
    }

    rendered_outputs = {
        "corpus": render_jsonl(corpus_records),
        "query_templates": render_jsonl(query_templates),
        "annotations": render_annotation_tsv(
            chunks=chunk_list,
            query_templates=query_templates,
        ),
        "manifest": json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }

    if not overwrite:
        existing_paths = [
            path
            for path in paths.values()
            if path.exists()
        ]

        if existing_paths:
            raise FileExistsError(
                "Benchmark export would overwrite existing files: "
                + ", ".join(
                    str(path)
                    for path in existing_paths
                )
            )

    for name, path in paths.items():
        _atomic_write_text(
            path,
            rendered_outputs[name],
            overwrite=overwrite,
        )

    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for block in iter(
            lambda: source_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an academic PDF as retrieval benchmark annotation tasks."
        )
    )
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--paper-id")
    parser.add_argument("--source-url")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Academic PDF does not exist: {pdf_path}"
        )

    paper_id = args.paper_id or pdf_path.stem
    chunking_config = {
        "target_tokens": 800,
        "overlap_tokens": 120,
        "min_chunk_tokens": 100,
        "encoding_name": "cl100k_base",
    }
    chunker = AcademicPaperChunker(**chunking_config)
    chunks = chunker.chunk_pdf(
        file_path=pdf_path,
        paper_id=paper_id,
        source_url=args.source_url,
    )

    paths = export_benchmark_artifacts(
        chunks=chunks,
        paper_id=paper_id,
        output_dir=args.output_dir,
        source_document=pdf_path.name,
        source_sha256=_sha256_file(pdf_path),
        source_url=args.source_url,
        overwrite=args.overwrite,
        chunking_config=chunking_config,
    )

    print(
        f"Exported {len(chunks)} chunks and "
        f"{len(REQUIRED_DIMENSIONS)} query templates."
    )

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
