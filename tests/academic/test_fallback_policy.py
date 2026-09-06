import pytest

from gpt_researcher.academic.fallback_policy import (
    AcademicSearchFallback,
)
from gpt_researcher.academic.schemas import (
    PaperMetadata,
)


class StubRetriever:
    """
    测试使用的虚拟检索器，不访问真实API。
    """

    def __init__(
        self,
        papers: list[PaperMetadata] | None = None,
        error: RuntimeError | None = None,
    ) -> None:
        self.papers = papers or []
        self.error = error
        self.calls = 0

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[PaperMetadata]:
        self.calls += 1

        if self.error:
            raise self.error

        return self.papers[:limit]


def make_paper(
    provider: str,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=f"{provider}-paper",
        title=f"{provider} example paper",
        retrieved_from=provider,
    )


@pytest.mark.asyncio
async def test_openalex_success_does_not_use_fallback() -> None:
    openalex = StubRetriever(
        papers=[make_paper("openalex")]
    )

    crossref = StubRetriever(
        papers=[make_paper("crossref")]
    )

    policy = AcademicSearchFallback(
        openalex_retriever=openalex,
        crossref_retriever=crossref,
    )

    result = await policy.search(
        "retrieval augmented generation"
    )

    assert result.provider_used == "openalex"
    assert result.fallback_used is False
    assert result.errors == []
    assert openalex.calls == 1
    assert crossref.calls == 0


@pytest.mark.asyncio
async def test_openalex_error_uses_crossref() -> None:
    openalex = StubRetriever(
        error=RuntimeError(
            "OpenAlex returned HTTP 401."
        )
    )

    crossref = StubRetriever(
        papers=[make_paper("crossref")]
    )

    policy = AcademicSearchFallback(
        openalex_retriever=openalex,
        crossref_retriever=crossref,
    )

    result = await policy.search(
        "retrieval augmented generation"
    )

    assert result.provider_used == "crossref"
    assert result.fallback_used is True
    assert len(result.papers) == 1
    assert len(result.errors) == 1
    assert "401" in result.errors[0]


@pytest.mark.asyncio
async def test_empty_openalex_result_uses_crossref() -> None:
    openalex = StubRetriever(
        papers=[]
    )

    crossref = StubRetriever(
        papers=[make_paper("crossref")]
    )

    policy = AcademicSearchFallback(
        openalex_retriever=openalex,
        crossref_retriever=crossref,
    )

    result = await policy.search(
        "retrieval augmented generation"
    )

    assert result.provider_used == "crossref"
    assert result.fallback_used is True
    assert result.errors == [
        "OpenAlex returned no results."
    ]


@pytest.mark.asyncio
async def test_all_retrievers_fail() -> None:
    openalex = StubRetriever(
        error=RuntimeError(
            "OpenAlex timeout."
        )
    )

    crossref = StubRetriever(
        error=RuntimeError(
            "Crossref timeout."
        )
    )

    policy = AcademicSearchFallback(
        openalex_retriever=openalex,
        crossref_retriever=crossref,
    )

    with pytest.raises(
        RuntimeError,
        match="All academic retrievers failed",
    ):
        await policy.search(
            "retrieval augmented generation"
        )