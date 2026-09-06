from typing import Protocol

from gpt_researcher.academic.schemas import (
    AcademicSearchResult,
    PaperMetadata,
)
from gpt_researcher.academic.source_retriever import (
    CrossrefRetriever,
    OpenAlexRetriever,
)
from gpt_researcher.academic.paper_identity import deduplicate_papers


class AcademicRetriever(Protocol):
    """
    规定学术检索器必须提供的统一接口。
    """

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[PaperMetadata]:
        ...


class AcademicSearchFallback:
    """
    优先使用OpenAlex，失败或无结果时切换到Crossref。
    """

    def __init__(
        self,
        openalex_retriever: AcademicRetriever | None = None,
        crossref_retriever: AcademicRetriever | None = None,
    ) -> None:
        self.openalex = (
            openalex_retriever
            or OpenAlexRetriever()
        )

        self.crossref = (
            crossref_retriever
            or CrossrefRetriever()
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> AcademicSearchResult:
        query = query.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if limit < 1:
            raise ValueError(
                "Search limit must be at least 1."
            )

        errors: list[str] = []

        try:
            openalex_papers = (
                await self.openalex.search(
                    query=query,
                    limit=limit,
                )
            )

            if openalex_papers:
                return AcademicSearchResult(
                    query=query,
                    papers=deduplicate_papers(openalex_papers)[:limit],
                    provider_used="openalex",
                    fallback_used=False,
                    errors=errors,
                )

            errors.append(
                "OpenAlex returned no results."
            )

        except RuntimeError as exc:
            errors.append(
                f"OpenAlex failed: {exc}"
            )

        try:
            crossref_papers = (
                await self.crossref.search(
                    query=query,
                    limit=limit,
                )
            )

            if crossref_papers:
                return AcademicSearchResult(
                    query=query,
                    papers=deduplicate_papers(crossref_papers)[:limit],
                    provider_used="crossref",
                    fallback_used=True,
                    errors=errors,
                )

            errors.append(
                "Crossref returned no results."
            )

        except RuntimeError as exc:
            errors.append(
                f"Crossref failed: {exc}"
            )

        error_summary = " | ".join(errors)

        raise RuntimeError(
            "All academic retrievers failed. "
            f"{error_summary}"
        )
