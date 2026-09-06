from __future__ import annotations

import json

import pytest

from backend.academic_api.services.evaluation_runner import (
    load_checkpoint_external_evidence,
    write_checkpoint,
)
from gpt_researcher.academic.external_evidence import (
    ExternalScholarlyEvidenceRetriever,
)
from gpt_researcher.academic.schemas import (
    AcademicSearchResult,
    ExternalEvidenceSummary,
    PaperMetadata,
)


class StaticSearchClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int = 10):
        self.calls.append((query, limit))
        if self.error is not None:
            raise self.error
        return self.result


def make_paper(paper_id: str, title: str, year: int | None) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        year=year,
        abstract="Bounded external abstract.",
        retrieved_from="openalex",
    )


@pytest.mark.asyncio
async def test_external_retrieval_excludes_target_and_marks_time_eligibility() -> None:
    client = StaticSearchClient(
        AcademicSearchResult(
            query="Target Paper",
            provider_used="openalex",
            papers=[
                make_paper("target", "Target Paper", 2020),
                make_paper("prior", "Prior Work", 2019),
                make_paper("future", "Future Work", 2022),
                make_paper("unknown", "Unknown Year", None),
            ],
        )
    )
    retriever = ExternalScholarlyEvidenceRetriever(client)

    result = await retriever.retrieve(
        paper_id="target",
        title="Target Paper",
        limit=3,
    )

    assert client.calls == [("Target Paper", 4)]
    assert result.status == "available"
    assert result.target_year == 2020
    assert [record.paper.paper_id for record in result.records] == [
        "prior",
        "future",
        "unknown",
    ]
    assert [record.eligible_for_novelty for record in result.records] == [
        True,
        False,
        False,
    ]


@pytest.mark.asyncio
async def test_external_retrieval_failure_is_non_fatal_and_sanitized() -> None:
    client = StaticSearchClient(error=RuntimeError("secret provider response"))
    retriever = ExternalScholarlyEvidenceRetriever(client)

    result = await retriever.retrieve(
        paper_id="paper-1",
        title="A title",
    )

    assert result.status == "unavailable"
    assert result.records == []
    assert "secret provider response" not in json.dumps(result.model_dump())
    assert "error_type=RuntimeError" in result.warnings


def test_checkpoint_restores_frozen_external_evidence(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    summary = ExternalEvidenceSummary(
        status="no_results",
        query="Target Paper",
        warnings=["No eligible records were found."],
    )
    write_checkpoint(
        path,
        {
            "paper_id": "paper-1",
            "completed": False,
            "dimension_results": [],
            "external_evidence": summary.model_dump(mode="json"),
        },
    )

    assert load_checkpoint_external_evidence(path, "paper-1") == summary
