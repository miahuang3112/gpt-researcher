from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evals.academic_retrieval.dataset import (
    load_corpus_jsonl,
)
from evals.academic_retrieval.export_pdf_benchmark import (
    build_query_templates,
    export_benchmark_artifacts,
)
from gpt_researcher.academic.paper_evaluator import (
    REQUIRED_DIMENSIONS,
)
from gpt_researcher.academic.schemas import EvidenceChunk


@pytest.fixture
def chunks() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id="paper-1-p1-c1",
            paper_id="paper-1",
            text=(
                "The research objective is to improve evidence retrieval."
            ),
            section="Introduction",
            page=1,
            source_url="https://example.org/paper-1",
            fused_score=0.9,
        ),
        EvidenceChunk(
            chunk_id="paper-1-p2-c2",
            paper_id="paper-1",
            text=(
                "Random seeds, hyperparameters and hardware are reported."
            ),
            section="Methods",
            page=2,
            source_url="https://example.org/paper-1",
            bm25_score=4.2,
        ),
    ]


def test_build_query_templates_covers_dimensions_in_scoring_order() -> None:
    templates = build_query_templates("paper-1")

    assert [
        template["dimension"]
        for template in templates
    ] == list(REQUIRED_DIMENSIONS)
    assert len(
        {
            template["query_id"]
            for template in templates
        }
    ) == len(REQUIRED_DIMENSIONS)
    assert all(
        template["query"].strip()
        for template in templates
    )


def test_export_writes_deterministic_annotation_artifacts(
    tmp_path: Path,
    chunks: list[EvidenceChunk],
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    arguments = {
        "chunks": chunks,
        "paper_id": "paper-1",
        "source_document": "paper.pdf",
        "source_sha256": "abc123",
        "source_url": "https://example.org/paper-1",
        "chunking_config": {
            "target_tokens": 800,
            "overlap_tokens": 120,
        },
    }

    first_paths = export_benchmark_artifacts(
        output_dir=first_dir,
        **arguments,
    )
    second_paths = export_benchmark_artifacts(
        output_dir=second_dir,
        **arguments,
    )

    assert set(first_paths) == {
        "corpus",
        "query_templates",
        "annotations",
        "manifest",
    }

    for name in first_paths:
        assert first_paths[name].read_bytes() == second_paths[name].read_bytes()


def test_exported_corpus_preserves_metadata_but_not_runtime_scores(
    tmp_path: Path,
    chunks: list[EvidenceChunk],
) -> None:
    paths = export_benchmark_artifacts(
        chunks=chunks,
        paper_id="paper-1",
        output_dir=tmp_path,
        source_document="paper.pdf",
        source_sha256="abc123",
    )

    loaded_chunks = load_corpus_jsonl(paths["corpus"])

    assert loaded_chunks[0].section == "Introduction"
    assert loaded_chunks[0].page == 1
    assert loaded_chunks[0].source_url == "https://example.org/paper-1"
    assert loaded_chunks[0].fused_score is None
    assert loaded_chunks[1].bm25_score is None


def test_annotation_tsv_contains_every_query_chunk_pair(
    tmp_path: Path,
    chunks: list[EvidenceChunk],
) -> None:
    paths = export_benchmark_artifacts(
        chunks=chunks,
        paper_id="paper-1",
        output_dir=tmp_path,
        source_document="paper.pdf",
        source_sha256="abc123",
    )

    with paths["annotations"].open(
        encoding="utf-8",
        newline="",
    ) as annotation_file:
        rows = list(
            csv.DictReader(
                annotation_file,
                delimiter="\t",
            )
        )

    assert len(rows) == len(REQUIRED_DIMENSIONS) * len(chunks)
    assert all(row["relevance_grade"] == "" for row in rows)
    assert all(row["paper_id"] == "paper-1" for row in rows)
    assert {
        row["chunk_id"]
        for row in rows
    } == {
        chunk.chunk_id
        for chunk in chunks
    }


def test_manifest_describes_frozen_export(
    tmp_path: Path,
    chunks: list[EvidenceChunk],
) -> None:
    paths = export_benchmark_artifacts(
        chunks=chunks,
        paper_id="paper-1",
        output_dir=tmp_path,
        source_document="paper.pdf",
        source_sha256="abc123",
    )

    manifest = json.loads(
        paths["manifest"].read_text(encoding="utf-8")
    )

    assert manifest["paper_id"] == "paper-1"
    assert manifest["source_document"] == "paper.pdf"
    assert manifest["source_sha256"] == "abc123"
    assert manifest["chunk_count"] == 2
    assert manifest["query_count"] == len(REQUIRED_DIMENSIONS)
    assert manifest["annotation_task_count"] == (
        2 * len(REQUIRED_DIMENSIONS)
    )


def test_export_refuses_to_overwrite_existing_artifacts(
    tmp_path: Path,
    chunks: list[EvidenceChunk],
) -> None:
    arguments = {
        "chunks": chunks,
        "paper_id": "paper-1",
        "output_dir": tmp_path,
        "source_document": "paper.pdf",
        "source_sha256": "abc123",
    }
    export_benchmark_artifacts(**arguments)

    with pytest.raises(
        FileExistsError,
        match="would overwrite existing files",
    ):
        export_benchmark_artifacts(**arguments)


def test_export_rejects_chunks_from_another_paper(
    tmp_path: Path,
    chunks: list[EvidenceChunk],
) -> None:
    mismatched_chunks = [
        chunks[0],
        chunks[1].model_copy(
            update={"paper_id": "paper-2"}
        ),
    ]

    with pytest.raises(
        ValueError,
        match="every chunk must belong",
    ):
        export_benchmark_artifacts(
            chunks=mismatched_chunks,
            paper_id="paper-1",
            output_dir=tmp_path,
            source_document="paper.pdf",
            source_sha256="abc123",
        )
