import pytest

from gpt_researcher.academic.bm25_retriever import (
    BM25Retriever,
)
from gpt_researcher.academic.schemas import (
    EvidenceChunk,
)


@pytest.fixture
def evidence_chunks() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id="chunk-model",
            paper_id="paper-1",
            text=(
                "We evaluate Qwen3-Embedding-0.6B "
                "on the MTEB benchmark."
            ),
        ),
        EvidenceChunk(
            chunk_id="chunk-generic",
            paper_id="paper-2",
            text=(
                "Sentence embedding models are evaluated "
                "on semantic similarity tasks."
            ),
        ),
        EvidenceChunk(
            chunk_id="chunk-doi",
            paper_id="paper-3",
            text=(
                "The original study has DOI "
                "10.1234/example.2026."
            ),
        ),
        EvidenceChunk(
            chunk_id="chunk-bm25",
            paper_id="paper-4",
            text=(
                "BM25 performs lexical matching "
                "for document retrieval."
            ),
        ),
        EvidenceChunk(
            chunk_id="chunk-vision",
            paper_id="paper-5",
            text=(
                "A vision transformer is trained "
                "on the ImageNet dataset."
            ),
        ),
    ]


def test_tokenizer_preserves_technical_identifiers() -> None:
    tokens = BM25Retriever.tokenize(
        "Qwen3-Embedding-0.6B MTEB "
        "10.1234/Example.2026"
    )

    assert "qwen3-embedding-0.6b" in tokens
    assert "qwen3" in tokens
    assert "embedding" in tokens
    assert "mteb" in tokens
    assert "10.1234/example.2026" in tokens


def test_exact_model_name_is_ranked_first(
    evidence_chunks: list[EvidenceChunk],
) -> None:
    retriever = BM25Retriever(
        evidence_chunks
    )

    results = retriever.search(
        "Qwen3-Embedding-0.6B MTEB",
        top_k=3,
    )

    assert results[0].chunk_id == "chunk-model"
    assert results[0].bm25_score is not None


def test_exact_doi_is_ranked_first(
    evidence_chunks: list[EvidenceChunk],
) -> None:
    retriever = BM25Retriever(
        evidence_chunks
    )

    results = retriever.search(
        "10.1234/example.2026",
        top_k=3,
    )

    assert results[0].chunk_id == "chunk-doi"
    assert results[0].bm25_score is not None


def test_unknown_query_returns_no_results(
    evidence_chunks: list[EvidenceChunk],
) -> None:
    retriever = BM25Retriever(
        evidence_chunks
    )

    results = retriever.search(
        "completely-unknown-identifier"
    )

    assert results == []


def test_top_k_and_original_chunks_are_unchanged(
    evidence_chunks: list[EvidenceChunk],
) -> None:
    retriever = BM25Retriever(
        evidence_chunks
    )

    results = retriever.search(
        "document retrieval",
        top_k=2,
    )

    assert len(results) == 2

    # 返回的是附带BM25分数的副本。
    assert results[0].bm25_score is not None

    # 原始EvidenceChunk没有被直接修改。
    assert all(
        chunk.bm25_score is None
        for chunk in evidence_chunks
    )