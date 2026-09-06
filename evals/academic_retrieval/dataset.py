from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from gpt_researcher.academic.schemas import EvidenceChunk


class RetrievalDatasetError(ValueError):
    """A retrieval benchmark dataset is missing or malformed."""


class RetrievalBenchmarkQuery(BaseModel):
    """One query and its human-assigned chunk relevance judgements."""

    query_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    dimension: str | None = None
    relevance_by_chunk_id: dict[str, float]

    @model_validator(mode="after")
    def validate_relevance_judgements(
        self,
    ) -> "RetrievalBenchmarkQuery":
        if any(
            not chunk_id.strip()
            for chunk_id in self.relevance_by_chunk_id
        ):
            raise ValueError(
                "relevance chunk IDs must not be blank"
            )

        grades = self.relevance_by_chunk_id.values()

        if any(
            not math.isfinite(grade) or grade < 0
            for grade in grades
        ):
            raise ValueError(
                "relevance grades must be finite and non-negative"
            )

        if not any(
            grade > 0
            for grade in self.relevance_by_chunk_id.values()
        ):
            raise ValueError(
                "at least one chunk must have positive relevance"
            )

        return self


def _read_jsonl(path: str | Path) -> list[tuple[int, dict]]:
    resolved_path = Path(path).expanduser().resolve()

    if not resolved_path.is_file():
        raise RetrievalDatasetError(
            f"JSONL file does not exist: {resolved_path}"
        )

    records: list[tuple[int, dict]] = []

    for line_number, raw_line in enumerate(
        resolved_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalDatasetError(
                f"Invalid JSON at {resolved_path}:{line_number}: {exc.msg}"
            ) from exc

        if not isinstance(payload, dict):
            raise RetrievalDatasetError(
                f"Expected a JSON object at {resolved_path}:{line_number}"
            )

        records.append((line_number, payload))

    if not records:
        raise RetrievalDatasetError(
            f"JSONL file contains no records: {resolved_path}"
        )

    return records


def load_corpus_jsonl(
    path: str | Path,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    seen_chunk_ids: set[str] = set()

    for line_number, payload in _read_jsonl(path):
        try:
            chunk = EvidenceChunk.model_validate(payload)
        except ValidationError as exc:
            raise RetrievalDatasetError(
                f"Invalid corpus record on line {line_number}: {exc}"
            ) from exc

        if not chunk.chunk_id.strip():
            raise RetrievalDatasetError(
                f"Blank chunk_id on corpus line {line_number}"
            )

        if not chunk.paper_id.strip():
            raise RetrievalDatasetError(
                f"Blank paper_id on corpus line {line_number}"
            )

        if not chunk.text.strip():
            raise RetrievalDatasetError(
                f"Blank chunk text on corpus line {line_number}"
            )

        if chunk.chunk_id in seen_chunk_ids:
            raise RetrievalDatasetError(
                f"Duplicate corpus chunk_id: {chunk.chunk_id}"
            )

        seen_chunk_ids.add(chunk.chunk_id)
        chunks.append(chunk)

    return chunks


def load_queries_jsonl(
    path: str | Path,
) -> list[RetrievalBenchmarkQuery]:
    queries: list[RetrievalBenchmarkQuery] = []
    seen_query_ids: set[str] = set()

    for line_number, payload in _read_jsonl(path):
        try:
            query = RetrievalBenchmarkQuery.model_validate(
                payload
            )
        except ValidationError as exc:
            raise RetrievalDatasetError(
                f"Invalid query record on line {line_number}: {exc}"
            ) from exc

        if query.query_id in seen_query_ids:
            raise RetrievalDatasetError(
                f"Duplicate query_id: {query.query_id}"
            )

        seen_query_ids.add(query.query_id)
        queries.append(query)

    return queries


def validate_query_references(
    chunks: list[EvidenceChunk],
    queries: list[RetrievalBenchmarkQuery],
) -> None:
    chunk_paper_by_id = {
        chunk.chunk_id: chunk.paper_id
        for chunk in chunks
    }

    corpus_paper_ids = set(chunk_paper_by_id.values())

    for query in queries:
        if query.paper_id not in corpus_paper_ids:
            raise RetrievalDatasetError(
                f"Query {query.query_id} references unknown paper_id: "
                f"{query.paper_id}"
            )

        for chunk_id in query.relevance_by_chunk_id:
            chunk_paper_id = chunk_paper_by_id.get(chunk_id)

            if chunk_paper_id is None:
                raise RetrievalDatasetError(
                    f"Query {query.query_id} references unknown chunk_id: "
                    f"{chunk_id}"
                )

            if chunk_paper_id != query.paper_id:
                raise RetrievalDatasetError(
                    f"Query {query.query_id} references chunk {chunk_id} "
                    f"from paper {chunk_paper_id}, not {query.paper_id}"
                )
