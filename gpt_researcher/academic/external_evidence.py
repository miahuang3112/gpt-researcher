from __future__ import annotations

import hashlib
from typing import Protocol

from gpt_researcher.academic.paper_identity import normalize_title
from gpt_researcher.academic.schemas import (
    AcademicSearchResult,
    ExternalEvidenceSummary,
    ExternalScholarlyRecord,
    PaperMetadata,
)


class AcademicSearchClient(Protocol):
    async def search(self, query: str, limit: int = 10) -> AcademicSearchResult: ...


def _external_evidence_id(paper: PaperMetadata) -> str:
    identity = paper.paper_id or paper.doi or paper.title
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"external:{digest}"


class ExternalScholarlyEvidenceRetriever:
    """Resolve bounded prior-work metadata for two scoring dimensions."""

    def __init__(self, search_client: AcademicSearchClient) -> None:
        self.search_client = search_client

    async def retrieve(
        self,
        *,
        paper_id: str,
        title: str | None,
        limit: int = 5,
    ) -> ExternalEvidenceSummary:
        if limit < 1:
            raise ValueError("External evidence limit must be at least 1.")
        query = (title or paper_id).strip()
        if not query:
            return ExternalEvidenceSummary(
                status="no_results",
                warnings=["No title or paper identifier was available for search."],
            )

        try:
            result = await self.search_client.search(query=query, limit=limit + 1)
        except Exception as exc:
            return ExternalEvidenceSummary(
                status="unavailable",
                query=query,
                warnings=[
                    "External scholarly retrieval was unavailable; internal-evidence scoring continued.",
                    f"error_type={type(exc).__name__}",
                ],
            )

        normalized_target_title = normalize_title(title) if title else None
        target_year: int | None = None
        related: list[PaperMetadata] = []
        for paper in result.papers:
            same_id = paper.paper_id == paper_id
            same_title = bool(
                normalized_target_title
                and normalize_title(paper.title) == normalized_target_title
            )
            if same_id or same_title:
                if paper.year is not None:
                    target_year = paper.year
                continue
            related.append(paper)

        records: list[ExternalScholarlyRecord] = []
        for paper in related[:limit]:
            eligible = bool(
                target_year is not None
                and paper.year is not None
                and paper.year <= target_year
            )
            if target_year is None:
                temporal_note = (
                    "Target publication year is unknown; do not use this record "
                    "to judge novelty at publication time."
                )
            elif paper.year is None:
                temporal_note = (
                    "Record publication year is unknown; do not use this record "
                    "to judge novelty at publication time."
                )
            elif paper.year > target_year:
                temporal_note = (
                    f"Published after the target ({paper.year} > {target_year}); "
                    "not eligible as prior work."
                )
            else:
                temporal_note = (
                    f"Published no later than the target ({paper.year} <= "
                    f"{target_year}); eligible as candidate prior work."
                )
            records.append(
                ExternalScholarlyRecord(
                    external_evidence_id=_external_evidence_id(paper),
                    paper=paper,
                    eligible_for_novelty=eligible,
                    temporal_note=temporal_note,
                )
            )

        warnings = [
            "External search is discovery context, not an exhaustive literature review."
        ]
        if target_year is None:
            warnings.append(
                "The target publication year could not be verified; novelty use is restricted."
            )
        if result.errors:
            warnings.append(
                "The primary provider returned a partial/empty result and fallback routing was used."
            )
        return ExternalEvidenceSummary(
            status="available" if records else "no_results",
            query=query,
            provider_used=result.provider_used,
            fallback_used=result.fallback_used,
            target_year=target_year,
            records=records,
            warnings=warnings,
        )
