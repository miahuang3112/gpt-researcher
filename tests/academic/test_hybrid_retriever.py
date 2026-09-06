from collections.abc import Sequence

import pytest

from gpt_researcher.academic.hybrid_retriever import (
    HybridRetrievalError,
    HybridRetriever,
)
from gpt_researcher.academic.schemas import EvidenceChunk


class FakeBM25Retriever:
    def __init__(
        self,
        results: Sequence[EvidenceChunk],
    ) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeVectorRetriever:
    def __init__(
        self,
        results: Sequence[EvidenceChunk],
    ) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FailingBM25Retriever:
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        raise RuntimeError("BM25 unavailable")


class FailingVectorRetriever:
    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        raise RuntimeError("vector unavailable")


@pytest.fixture
def chunks() -> dict[str, EvidenceChunk]:
    return {
        "a": EvidenceChunk(
            chunk_id="a",
            paper_id="paper-1",
            text="Qwen3-Embedding-0.6B evaluation results.",
        ),
        "b": EvidenceChunk(
            chunk_id="b",
            paper_id="paper-1",
            text="The embedding model retrieves relevant evidence.",
        ),
        "c": EvidenceChunk(
            chunk_id="c",
            paper_id="paper-2",
            text="Semantic retrieval finds conceptually similar text.",
        ),
    }


@pytest.fixture
def bm25_results(
    chunks: dict[str, EvidenceChunk],
) -> list[EvidenceChunk]:
    return [
        chunks["a"].model_copy(
            update={"bm25_score": 12.0}
        ),
        chunks["b"].model_copy(
            update={"bm25_score": 8.0}
        ),
    ]


@pytest.fixture
def vector_results(
    chunks: dict[str, EvidenceChunk],
) -> list[EvidenceChunk]:
    return [
        chunks["c"].model_copy(
            update={"vector_score": 0.95}
        ),
        chunks["b"].model_copy(
            update={"vector_score": 0.85}
        ),
    ]


@pytest.mark.asyncio
async def test_chunk_found_by_both_retrievers_is_ranked_first(
    bm25_results: list[EvidenceChunk],
    vector_results: list[EvidenceChunk],
) -> None:
    retriever = HybridRetriever(
        FakeBM25Retriever(bm25_results),
        FakeVectorRetriever(vector_results),
    )

    results = await retriever.search(
        "embedding retrieval",
        top_k=3,
    )

    assert results[0].chunk_id == "b"
    assert results[0].fused_score is not None
    assert results[0].fused_score > results[1].fused_score


@pytest.mark.asyncio
async def test_scores_from_both_retrievers_are_preserved(
    bm25_results: list[EvidenceChunk],
    vector_results: list[EvidenceChunk],
) -> None:
    retriever = HybridRetriever(
        FakeBM25Retriever(bm25_results),
        FakeVectorRetriever(vector_results),
    )

    results = await retriever.search(
        "embedding retrieval",
        top_k=3,
    )

    shared_result = results[0]

    assert shared_result.chunk_id == "b"
    assert shared_result.bm25_score == pytest.approx(8.0)
    assert shared_result.vector_score == pytest.approx(0.85)
    assert shared_result.fused_score == pytest.approx(
        2.0 / 62.0
    )


@pytest.mark.asyncio
async def test_fused_results_do_not_contain_duplicates(
    bm25_results: list[EvidenceChunk],
    vector_results: list[EvidenceChunk],
) -> None:
    retriever = HybridRetriever(
        FakeBM25Retriever(bm25_results),
        FakeVectorRetriever(vector_results),
    )

    results = await retriever.search(
        "embedding retrieval",
        top_k=3,
    )

    chunk_ids = [result.chunk_id for result in results]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_ids.count("b") == 1


@pytest.mark.asyncio
async def test_both_retrievers_receive_candidate_limit(
    bm25_results: list[EvidenceChunk],
    vector_results: list[EvidenceChunk],
) -> None:
    bm25 = FakeBM25Retriever(bm25_results)
    vector = FakeVectorRetriever(vector_results)

    retriever = HybridRetriever(bm25, vector)

    await retriever.search(
        "embedding retrieval",
        top_k=2,
        candidate_top_k=8,
    )

    assert bm25.calls == [("embedding retrieval", 8)]
    assert vector.calls == [("embedding retrieval", 8)]


@pytest.mark.asyncio
async def test_original_chunks_are_not_modified(
    chunks: dict[str, EvidenceChunk],
    bm25_results: list[EvidenceChunk],
    vector_results: list[EvidenceChunk],
) -> None:
    retriever = HybridRetriever(
        FakeBM25Retriever(bm25_results),
        FakeVectorRetriever(vector_results),
    )

    results = await retriever.search(
        "embedding retrieval",
        top_k=3,
    )

    assert chunks["b"].bm25_score is None
    assert chunks["b"].vector_score is None
    assert chunks["b"].fused_score is None
    assert results[0] is not chunks["b"]


@pytest.mark.asyncio
async def test_vector_failure_falls_back_to_bm25(
    bm25_results: list[EvidenceChunk],
) -> None:
    retriever = HybridRetriever(
        FakeBM25Retriever(bm25_results),
        FailingVectorRetriever(),
    )

    results = await retriever.search(
        "Qwen3 embedding",
        top_k=2,
    )

    assert [result.chunk_id for result in results] == ["a", "b"]
    assert results[0].bm25_score == pytest.approx(12.0)
    assert results[0].vector_score is None
    assert results[0].fused_score == pytest.approx(1.0 / 61.0)


@pytest.mark.asyncio
async def test_bm25_failure_falls_back_to_vector(
    vector_results: list[EvidenceChunk],
) -> None:
    retriever = HybridRetriever(
        FailingBM25Retriever(),
        FakeVectorRetriever(vector_results),
    )

    results = await retriever.search(
        "semantic retrieval",
        top_k=2,
    )

    assert [result.chunk_id for result in results] == ["c", "b"]
    assert results[0].bm25_score is None
    assert results[0].vector_score == pytest.approx(0.95)
    assert results[0].fused_score == pytest.approx(1.0 / 61.0)


@pytest.mark.asyncio
async def test_both_retriever_failures_raise_aggregated_error() -> None:
    retriever = HybridRetriever(
        FailingBM25Retriever(),
        FailingVectorRetriever(),
    )

    with pytest.raises(
        HybridRetrievalError,
        match=(
            "BM25 and vector retrieval both failed.*"
            "BM25 unavailable.*vector unavailable"
        ),
    ):
        await retriever.search("embedding retrieval")


@pytest.mark.asyncio
async def test_empty_vector_results_keep_bm25_results(
    bm25_results: list[EvidenceChunk],
) -> None:
    retriever = HybridRetriever(
        FakeBM25Retriever(bm25_results),
        FakeVectorRetriever([]),
    )

    results = await retriever.search(
        "embedding retrieval",
        top_k=2,
    )

    assert [result.chunk_id for result in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_invalid_search_arguments_are_rejected(
    bm25_results: list[EvidenceChunk],
    vector_results: list[EvidenceChunk],
) -> None:
    retriever = HybridRetriever(
        FakeBM25Retriever(bm25_results),
        FakeVectorRetriever(vector_results),
    )

    with pytest.raises(
        ValueError,
        match="query must not be blank",
    ):
        await retriever.search("   ")

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        await retriever.search(
            "embedding",
            top_k=0,
        )

    with pytest.raises(
        ValueError,
        match="candidate_top_k",
    ):
        await retriever.search(
            "embedding",
            top_k=5,
            candidate_top_k=3,
        )


@pytest.mark.asyncio
async def test_optional_source_cap_preserves_rrf_order_across_sources() -> None:
    chunks = [
        EvidenceChunk(chunk_id="p1-a", paper_id="paper-1", text="a"),
        EvidenceChunk(chunk_id="p1-b", paper_id="paper-1", text="b"),
        EvidenceChunk(chunk_id="p2-a", paper_id="paper-2", text="c"),
        EvidenceChunk(chunk_id="p3-a", paper_id="paper-3", text="d"),
    ]
    retriever = HybridRetriever(
        FakeBM25Retriever(chunks),
        FakeVectorRetriever([]),
        max_chunks_per_source=1,
    )

    results = await retriever.search("diverse evidence", top_k=3)

    assert [item.chunk_id for item in results] == ["p1-a", "p2-a", "p3-a"]


@pytest.mark.asyncio
async def test_source_cap_is_disabled_by_default() -> None:
    chunks = [
        EvidenceChunk(chunk_id="a", paper_id="paper-1", text="a"),
        EvidenceChunk(chunk_id="b", paper_id="paper-1", text="b"),
    ]
    retriever = HybridRetriever(
        FakeBM25Retriever(chunks),
        FakeVectorRetriever([]),
    )

    results = await retriever.search("single paper", top_k=2)

    assert [item.chunk_id for item in results] == ["a", "b"]


def test_invalid_source_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_chunks_per_source"):
        HybridRetriever(
            FakeBM25Retriever([]),
            FakeVectorRetriever([]),
            max_chunks_per_source=0,
        )
