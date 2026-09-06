from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evals.academic_retrieval.compile_annotations import (
    compile_annotation_tsv,
    write_compiled_queries,
)
from evals.academic_retrieval.dataset import (
    RetrievalDatasetError,
    load_queries_jsonl,
)
from evals.academic_retrieval.export_pdf_benchmark import (
    export_benchmark_artifacts,
)
from gpt_researcher.academic.paper_evaluator import (
    REQUIRED_DIMENSIONS,
)
from gpt_researcher.academic.schemas import EvidenceChunk


def _export_annotation_fixture(
    tmp_path: Path,
) -> dict[str, Path]:
    chunks = [
        EvidenceChunk(
            chunk_id="c1",
            paper_id="paper-1",
            text="The research question and motivation are stated.",
        ),
        EvidenceChunk(
            chunk_id="c2",
            paper_id="paper-1",
            text="Experiments report datasets and baselines.",
        ),
    ]

    return export_benchmark_artifacts(
        chunks=chunks,
        paper_id="paper-1",
        output_dir=tmp_path,
        source_document="paper.pdf",
        source_sha256="abc123",
    )


def _read_annotation_rows(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _write_annotation_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _complete_all_annotations(
    path: Path,
) -> None:
    fieldnames, rows = _read_annotation_rows(path)
    seen_query_ids: set[str] = set()

    for row in rows:
        query_id = row["query_id"]
        row["relevance_grade"] = (
            "2" if query_id not in seen_query_ids else "0"
        )
        seen_query_ids.add(query_id)

    _write_annotation_rows(path, fieldnames, rows)


def test_completed_annotations_compile_to_benchmark_queries(
    tmp_path: Path,
) -> None:
    paths = _export_annotation_fixture(tmp_path)
    _complete_all_annotations(paths["annotations"])

    queries = compile_annotation_tsv(
        annotation_path=paths["annotations"],
        corpus_path=paths["corpus"],
    )
    output_path = write_compiled_queries(
        queries=queries,
        output_path=tmp_path / "queries.jsonl",
    )
    loaded_queries = load_queries_jsonl(output_path)

    assert len(loaded_queries) == len(REQUIRED_DIMENSIONS)
    assert all(
        set(query.relevance_by_chunk_id) == {"c1", "c2"}
        for query in loaded_queries
    )
    assert all(
        sum(
            grade > 0
            for grade in query.relevance_by_chunk_id.values()
        )
        == 1
        for query in loaded_queries
    )


def test_blank_grade_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _export_annotation_fixture(tmp_path)

    with pytest.raises(
        RetrievalDatasetError,
        match="relevance_grade must be 0, 1 or 2",
    ):
        compile_annotation_tsv(
            annotation_path=paths["annotations"],
            corpus_path=paths["corpus"],
        )


def test_incomplete_query_chunk_matrix_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _export_annotation_fixture(tmp_path)
    _complete_all_annotations(paths["annotations"])
    fieldnames, rows = _read_annotation_rows(
        paths["annotations"]
    )
    rows.pop()
    _write_annotation_rows(
        paths["annotations"],
        fieldnames,
        rows,
    )

    with pytest.raises(
        RetrievalDatasetError,
        match="Incomplete annotations",
    ):
        compile_annotation_tsv(
            annotation_path=paths["annotations"],
            corpus_path=paths["corpus"],
        )


def test_duplicate_query_chunk_pair_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _export_annotation_fixture(tmp_path)
    _complete_all_annotations(paths["annotations"])
    fieldnames, rows = _read_annotation_rows(
        paths["annotations"]
    )
    rows.append(dict(rows[0]))
    _write_annotation_rows(
        paths["annotations"],
        fieldnames,
        rows,
    )

    with pytest.raises(
        RetrievalDatasetError,
        match="Duplicate query/chunk annotation pair",
    ):
        compile_annotation_tsv(
            annotation_path=paths["annotations"],
            corpus_path=paths["corpus"],
        )


def test_query_without_positive_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _export_annotation_fixture(tmp_path)
    fieldnames, rows = _read_annotation_rows(
        paths["annotations"]
    )

    for row in rows:
        row["relevance_grade"] = "0"

    _write_annotation_rows(
        paths["annotations"],
        fieldnames,
        rows,
    )

    with pytest.raises(
        RetrievalDatasetError,
        match="positive relevance",
    ):
        compile_annotation_tsv(
            annotation_path=paths["annotations"],
            corpus_path=paths["corpus"],
        )


def test_compiled_output_refuses_implicit_overwrite(
    tmp_path: Path,
) -> None:
    paths = _export_annotation_fixture(tmp_path / "source")
    _complete_all_annotations(paths["annotations"])
    queries = compile_annotation_tsv(
        annotation_path=paths["annotations"],
        corpus_path=paths["corpus"],
    )
    output_path = tmp_path / "queries.jsonl"
    write_compiled_queries(
        queries=queries,
        output_path=output_path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        write_compiled_queries(
            queries=queries,
            output_path=output_path,
        )
