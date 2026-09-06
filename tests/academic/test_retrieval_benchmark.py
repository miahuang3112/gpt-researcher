from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from evals.academic_retrieval.dataset import (
    RetrievalBenchmarkQuery,
    RetrievalDatasetError,
    load_corpus_jsonl,
    load_queries_jsonl,
    validate_query_references,
)
from evals.academic_retrieval.run_benchmark import (
    _build_provenance,
    run_benchmark,
)
from gpt_researcher.academic.schemas import EvidenceChunk


class MappingEmbeddingClient:
    def __init__(
        self,
        vectors_by_text: dict[str, list[float]],
    ) -> None:
        self.vectors_by_text = vectors_by_text

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [
            self.vectors_by_text[text]
            for text in texts
        ]


class FailingEmbeddingClient:
    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")


def test_benchmark_provenance_records_inputs_and_label_origin(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    labels_path = tmp_path / "labels.json"
    manifest_path = tmp_path / "manifest.json"
    corpus_path.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    queries_path.write_text('{"query_id":"q1"}\n', encoding="utf-8")
    labels_path.write_text(
        json.dumps(
            {
                "annotation_type": "AI pre-annotation",
                "annotator": "ChatGPT",
                "model_identifier": "test-model",
                "generated_at": "2026-09-01T00:00:00+00:00",
                "integrity_warning": "not human gold labels",
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "paper_id": "paper-1",
                "source_document": "paper.pdf",
                "source_sha256": "source-hash",
                "chunking": {"target_tokens": 800},
            }
        ),
        encoding="utf-8",
    )

    provenance = _build_provenance(
        corpus_path=corpus_path,
        queries_path=queries_path,
        embedding_model="embedding-test",
        label_provenance_path=labels_path,
        dataset_manifest_path=manifest_path,
    )

    assert provenance["embedding"] == {
        "provider": "ollama",
        "model": "embedding-test",
    }
    assert provenance["labels"]["annotation_type"] == "AI pre-annotation"
    assert provenance["labels"]["integrity_warning"] == "not human gold labels"
    assert provenance["dataset"]["source_sha256"] == "source-hash"
    assert len(provenance["corpus"]["sha256"]) == 64
    assert len(provenance["queries"]["sha256"]) == 64


@pytest.fixture
def benchmark_chunks() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id="c1",
            paper_id="paper-1",
            text="Qwen3-Embedding benchmark results.",
            section="Experiments",
            page=1,
        ),
        EvidenceChunk(
            chunk_id="c2",
            paper_id="paper-1",
            text="The method finds conceptually related passages.",
            section="Methods",
            page=2,
        ),
        EvidenceChunk(
            chunk_id="c3",
            paper_id="paper-1",
            text="Cooking recipes and ingredients.",
            section="Appendix",
            page=3,
        ),
    ]


@pytest.fixture
def benchmark_queries() -> list[RetrievalBenchmarkQuery]:
    return [
        RetrievalBenchmarkQuery(
            query_id="exact-model",
            paper_id="paper-1",
            query="Qwen3-Embedding",
            dimension="baseline_quality",
            relevance_by_chunk_id={"c1": 2.0},
        ),
        RetrievalBenchmarkQuery(
            query_id="semantic-paraphrase",
            paper_id="paper-1",
            query="semantic evidence retrieval",
            dimension="methodological_rigor",
            relevance_by_chunk_id={"c2": 2.0},
        ),
    ]


@pytest.fixture
def embedding_client(
    benchmark_chunks: list[EvidenceChunk],
    benchmark_queries: list[RetrievalBenchmarkQuery],
) -> MappingEmbeddingClient:
    return MappingEmbeddingClient(
        {
            benchmark_chunks[0].text: [1.0, 0.0, 0.0],
            benchmark_chunks[1].text: [0.0, 1.0, 0.0],
            benchmark_chunks[2].text: [0.0, 0.0, 1.0],
            benchmark_queries[0].query: [1.0, 0.0, 0.0],
            benchmark_queries[1].query: [0.0, 1.0, 0.0],
        }
    )


def test_jsonl_loaders_preserve_corpus_and_qrels(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"

    corpus_path.write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "paper_id": "paper-1",
                "text": "Reproducibility evidence.",
                "section": "Methods",
                "page": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "paper_id": "paper-1",
                "query": "Are implementation details provided?",
                "dimension": "reproducibility",
                "relevance_by_chunk_id": {"c1": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    chunks = load_corpus_jsonl(corpus_path)
    queries = load_queries_jsonl(queries_path)
    validate_query_references(chunks, queries)

    assert chunks[0].section == "Methods"
    assert chunks[0].page == 2
    assert queries[0].dimension == "reproducibility"
    assert queries[0].relevance_by_chunk_id == {"c1": 2.0}


def test_query_loader_rejects_no_positive_relevance(
    tmp_path,
) -> None:
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "paper_id": "paper-1",
                "query": "A query",
                "relevance_by_chunk_id": {"c1": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RetrievalDatasetError,
        match="positive relevance",
    ):
        load_queries_jsonl(queries_path)


def test_reference_validation_rejects_cross_paper_qrels(
    benchmark_chunks: list[EvidenceChunk],
) -> None:
    query = RetrievalBenchmarkQuery(
        query_id="q1",
        paper_id="paper-2",
        query="A query",
        relevance_by_chunk_id={"c1": 1.0},
    )
    other_paper_chunk = EvidenceChunk(
        chunk_id="other",
        paper_id="paper-2",
        text="Other paper evidence.",
    )

    with pytest.raises(
        RetrievalDatasetError,
        match="from paper paper-1, not paper-2",
    ):
        validate_query_references(
            benchmark_chunks + [other_paper_chunk],
            [query],
        )


@pytest.mark.asyncio
async def test_benchmark_compares_all_three_retrievers(
    benchmark_chunks: list[EvidenceChunk],
    benchmark_queries: list[RetrievalBenchmarkQuery],
    embedding_client: MappingEmbeddingClient,
) -> None:
    report = await run_benchmark(
        chunks=benchmark_chunks,
        queries=benchmark_queries,
        embedding_client=embedding_client,
        cutoffs=(1, 2),
        candidate_top_k=3,
    )

    assert set(report["summary"]) == {
        "bm25",
        "vector",
        "hybrid",
    }
    assert report["summary"]["vector"]["recall_at_k"]["1"] == 1.0
    assert report["summary"]["hybrid"]["query_count"] == 2
    assert report["summary"]["hybrid"]["failure_count"] == 0
    assert len(report["queries"]) == 2


@pytest.mark.asyncio
async def test_embedding_failure_is_counted_and_hybrid_falls_back(
    benchmark_chunks: list[EvidenceChunk],
) -> None:
    queries = [
        RetrievalBenchmarkQuery(
            query_id="exact-model",
            paper_id="paper-1",
            query="Qwen3-Embedding",
            relevance_by_chunk_id={"c1": 2.0},
        )
    ]

    report = await run_benchmark(
        chunks=benchmark_chunks,
        queries=queries,
        embedding_client=FailingEmbeddingClient(),
        cutoffs=(1,),
        candidate_top_k=2,
    )

    assert report["summary"]["vector"]["failure_count"] == 1
    assert report["summary"]["vector"]["recall_at_k"]["1"] == 0.0
    assert report["summary"]["hybrid"]["failure_count"] == 0
    assert report["summary"]["hybrid"]["recall_at_k"]["1"] == 1.0


@pytest.mark.asyncio
async def test_benchmark_rejects_candidate_depth_below_cutoff(
    benchmark_chunks: list[EvidenceChunk],
    benchmark_queries: list[RetrievalBenchmarkQuery],
    embedding_client: MappingEmbeddingClient,
) -> None:
    with pytest.raises(
        ValueError,
        match="largest cutoff",
    ):
        await run_benchmark(
            chunks=benchmark_chunks,
            queries=benchmark_queries,
            embedding_client=embedding_client,
            cutoffs=(1, 4),
            candidate_top_k=3,
        )
