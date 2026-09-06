from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .dataset import (
    RetrievalBenchmarkQuery,
    RetrievalDatasetError,
    load_corpus_jsonl,
    validate_query_references,
)
from .export_pdf_benchmark import render_jsonl


REQUIRED_ANNOTATION_FIELDS = {
    "query_id",
    "paper_id",
    "dimension",
    "query",
    "chunk_id",
    "relevance_grade",
}
ALLOWED_RELEVANCE_GRADES = {"0", "1", "2"}


def compile_annotation_tsv(
    *,
    annotation_path: str | Path,
    corpus_path: str | Path,
) -> list[RetrievalBenchmarkQuery]:
    resolved_annotation_path = (
        Path(annotation_path).expanduser().resolve()
    )

    if not resolved_annotation_path.is_file():
        raise RetrievalDatasetError(
            "Annotation TSV does not exist: "
            f"{resolved_annotation_path}"
        )

    chunks = load_corpus_jsonl(corpus_path)
    corpus_chunk_ids_by_paper: dict[str, set[str]] = defaultdict(set)

    for chunk in chunks:
        corpus_chunk_ids_by_paper[chunk.paper_id].add(
            chunk.chunk_id
        )

    query_metadata: dict[str, dict[str, str]] = {}
    relevance_by_query: dict[str, dict[str, float]] = {}

    with resolved_annotation_path.open(
        encoding="utf-8",
        newline="",
    ) as annotation_file:
        reader = csv.DictReader(
            annotation_file,
            delimiter="\t",
        )

        fieldnames = set(reader.fieldnames or [])
        missing_fields = REQUIRED_ANNOTATION_FIELDS - fieldnames

        if missing_fields:
            raise RetrievalDatasetError(
                "Annotation TSV is missing required columns: "
                + ", ".join(sorted(missing_fields))
            )

        for row in reader:
            line_number = reader.line_num
            query_id = (row.get("query_id") or "").strip()
            paper_id = (row.get("paper_id") or "").strip()
            dimension = (row.get("dimension") or "").strip()
            query_text = (row.get("query") or "").strip()
            chunk_id = (row.get("chunk_id") or "").strip()
            raw_grade = (
                row.get("relevance_grade") or ""
            ).strip()

            if not all(
                (query_id, paper_id, query_text, chunk_id)
            ):
                raise RetrievalDatasetError(
                    "Blank query/chunk metadata on annotation line "
                    f"{line_number}"
                )

            if raw_grade not in ALLOWED_RELEVANCE_GRADES:
                raise RetrievalDatasetError(
                    "relevance_grade must be 0, 1 or 2 on annotation "
                    f"line {line_number}; found {raw_grade!r}"
                )

            metadata = {
                "paper_id": paper_id,
                "dimension": dimension,
                "query": query_text,
            }
            existing_metadata = query_metadata.get(query_id)

            if existing_metadata is None:
                query_metadata[query_id] = metadata
                relevance_by_query[query_id] = {}
            elif existing_metadata != metadata:
                raise RetrievalDatasetError(
                    "Inconsistent metadata for query_id "
                    f"{query_id} on annotation line {line_number}"
                )

            query_relevances = relevance_by_query[query_id]

            if chunk_id in query_relevances:
                raise RetrievalDatasetError(
                    "Duplicate query/chunk annotation pair: "
                    f"{query_id}/{chunk_id}"
                )

            query_relevances[chunk_id] = float(raw_grade)

    if not query_metadata:
        raise RetrievalDatasetError(
            "Annotation TSV contains no annotation rows"
        )

    compiled_queries: list[RetrievalBenchmarkQuery] = []

    for query_id, metadata in query_metadata.items():
        paper_id = metadata["paper_id"]
        expected_chunk_ids = corpus_chunk_ids_by_paper.get(
            paper_id
        )

        if expected_chunk_ids is None:
            raise RetrievalDatasetError(
                f"Query {query_id} references unknown paper_id: {paper_id}"
            )

        actual_chunk_ids = set(
            relevance_by_query[query_id]
        )
        missing_chunk_ids = sorted(
            expected_chunk_ids - actual_chunk_ids
        )
        extra_chunk_ids = sorted(
            actual_chunk_ids - expected_chunk_ids
        )

        if missing_chunk_ids or extra_chunk_ids:
            raise RetrievalDatasetError(
                f"Incomplete annotations for query {query_id}. "
                f"Missing={missing_chunk_ids}, extra={extra_chunk_ids}"
            )

        try:
            compiled_query = RetrievalBenchmarkQuery(
                query_id=query_id,
                paper_id=paper_id,
                dimension=metadata["dimension"] or None,
                query=metadata["query"],
                relevance_by_chunk_id=(
                    relevance_by_query[query_id]
                ),
            )
        except Exception as exc:
            raise RetrievalDatasetError(
                f"Invalid compiled annotations for query {query_id}: {exc}"
            ) from exc

        compiled_queries.append(compiled_query)

    validate_query_references(chunks, compiled_queries)
    return compiled_queries


def write_compiled_queries(
    *,
    queries: list[RetrievalBenchmarkQuery],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    resolved_output_path = Path(output_path).expanduser().resolve()

    if resolved_output_path.exists() and not overwrite:
        raise FileExistsError(
            "Compiled query output already exists; pass --overwrite "
            f"to replace it: {resolved_output_path}"
        )

    records = [
        query.model_dump(
            mode="json",
            exclude_none=True,
        )
        for query in queries
    ]
    rendered_queries = render_jsonl(records)
    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = resolved_output_path.with_name(
        f".{resolved_output_path.name}.tmp"
    )
    temporary_path.write_text(
        rendered_queries,
        encoding="utf-8",
    )
    temporary_path.replace(resolved_output_path)
    return resolved_output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate completed retrieval annotations and compile qrels."
        )
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    queries = compile_annotation_tsv(
        annotation_path=args.annotations,
        corpus_path=args.corpus,
    )
    output_path = write_compiled_queries(
        queries=queries,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    positive_judgement_count = sum(
        grade > 0
        for query in queries
        for grade in query.relevance_by_chunk_id.values()
    )

    print(
        f"Compiled {len(queries)} queries with "
        f"{positive_judgement_count} positive judgements."
    )
    print(f"queries: {output_path}")


if __name__ == "__main__":
    main()
