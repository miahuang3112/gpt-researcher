from collections.abc import Sequence

import pytest

from gpt_researcher.academic.schemas import EvidenceChunk
from gpt_researcher.academic.vector_retriever import VectorRetriever


class FakeEmbeddingClient:
    """
    单元测试使用的假向量客户端。

    它不会调用 Ollama，测试结果固定且可以重复。
    """

    def __init__(
        self,
        vectors: dict[str, list[float]],
    ) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        text_list = list(texts)
        self.calls.append(text_list)

        return [self.vectors[text] for text in text_list]


@pytest.fixture
def chunks() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id="rag",
            paper_id="paper-1",
            text=(
                "Retrieval-augmented generation allows language models "
                "to use external evidence."
            ),
        ),
        EvidenceChunk(
            chunk_id="vision",
            paper_id="paper-2",
            text=(
                "Vision transformers process images using "
                "patch embeddings."
            ),
        ),
        EvidenceChunk(
            chunk_id="finance",
            paper_id="paper-3",
            text=(
                "Discounted cash flow estimates the value "
                "of an investment."
            ),
        ),
    ]


@pytest.fixture
def vectors(
    chunks: list[EvidenceChunk],
) -> dict[str, list[float]]:
    return {
        chunks[0].text: [1.0, 0.0, 0.0],
        chunks[1].text: [0.0, 1.0, 0.0],
        chunks[2].text: [0.0, 0.0, 1.0],
        "How can a language model use external knowledge?": [
            0.9,
            0.1,
            0.0,
        ],
        "Which model processes images?": [
            0.0,
            1.0,
            0.0,
        ],
    }


@pytest.mark.asyncio
async def test_search_returns_most_similar_chunk_first(
    chunks: list[EvidenceChunk],
    vectors: dict[str, list[float]],
) -> None:
    client = FakeEmbeddingClient(vectors)
    retriever = VectorRetriever(chunks, client)

    results = await retriever.search(
        "How can a language model use external knowledge?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "rag"
    assert results[0].vector_score is not None
    assert results[0].vector_score > results[1].vector_score


@pytest.mark.asyncio
async def test_document_vectors_are_built_only_once(
    chunks: list[EvidenceChunk],
    vectors: dict[str, list[float]],
) -> None:
    client = FakeEmbeddingClient(vectors)
    retriever = VectorRetriever(chunks, client)

    await retriever.search(
        "How can a language model use external knowledge?"
    )
    await retriever.search(
        "Which model processes images?"
    )

    document_calls = [
        call
        for call in client.calls
        if len(call) == len(chunks)
    ]

    assert len(document_calls) == 1


@pytest.mark.asyncio
async def test_search_does_not_modify_original_chunks(
    chunks: list[EvidenceChunk],
    vectors: dict[str, list[float]],
) -> None:
    client = FakeEmbeddingClient(vectors)
    retriever = VectorRetriever(chunks, client)

    results = await retriever.search(
        "How can a language model use external knowledge?"
    )

    assert chunks[0].vector_score is None
    assert results[0].vector_score is not None
    assert results[0] is not chunks[0]


@pytest.mark.asyncio
async def test_dimension_mismatch_raises_error(
    chunks: list[EvidenceChunk],
) -> None:
    vectors = {
        chunks[0].text: [1.0, 0.0, 0.0],
        chunks[1].text: [0.0, 1.0, 0.0],
        chunks[2].text: [0.0, 0.0, 1.0],
        "dimension mismatch": [1.0, 0.0],
    }

    retriever = VectorRetriever(
        chunks,
        FakeEmbeddingClient(vectors),
    )

    with pytest.raises(
        RuntimeError,
        match="dimension does not match",
    ):
        await retriever.search("dimension mismatch")


@pytest.mark.asyncio
async def test_blank_query_is_rejected(
    chunks: list[EvidenceChunk],
    vectors: dict[str, list[float]],
) -> None:
    retriever = VectorRetriever(
        chunks,
        FakeEmbeddingClient(vectors),
    )

    with pytest.raises(
        ValueError,
        match="query must not be blank",
    ):
        await retriever.search("   ")