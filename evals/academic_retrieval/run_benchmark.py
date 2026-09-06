from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpt_researcher.academic.bm25_retriever import BM25Retriever
from gpt_researcher.academic.hybrid_retriever import HybridRetriever
from gpt_researcher.academic.schemas import EvidenceChunk
from gpt_researcher.academic.vector_retriever import (
    EmbeddingClient,
    OllamaEmbeddingClient,
    VectorRetriever,
)

from .dataset import (
    RetrievalBenchmarkQuery,
    load_corpus_jsonl,
    load_queries_jsonl,
    validate_query_references,
)
from .metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


SYSTEM_NAMES = ("bm25", "vector", "hybrid")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not load {description} JSON from {path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")

    return payload


def _build_provenance(
    *,
    corpus_path: Path,
    queries_path: Path,
    embedding_model: str,
    label_provenance_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "evals.academic_retrieval.run_benchmark",
        "corpus": {
            "path": str(corpus_path),
            "sha256": _sha256_file(corpus_path),
        },
        "queries": {
            "path": str(queries_path),
            "sha256": _sha256_file(queries_path),
        },
        "embedding": {
            "provider": "ollama",
            "model": embedding_model,
        },
    }

    if label_provenance_path is not None:
        label_metadata = _load_json_object(
            label_provenance_path,
            "label provenance",
        )
        provenance["labels"] = {
            "path": str(label_provenance_path),
            "sha256": _sha256_file(label_provenance_path),
            "annotation_type": label_metadata.get("annotation_type"),
            "annotator": label_metadata.get("annotator"),
            "model_identifier": label_metadata.get("model_identifier"),
            "generated_at": label_metadata.get("generated_at"),
            "integrity_warning": label_metadata.get("integrity_warning"),
        }

    if dataset_manifest_path is not None:
        manifest = _load_json_object(
            dataset_manifest_path,
            "dataset manifest",
        )
        provenance["dataset"] = {
            "path": str(dataset_manifest_path),
            "manifest_sha256": _sha256_file(dataset_manifest_path),
            "paper_id": manifest.get("paper_id"),
            "source_document": manifest.get("source_document"),
            "source_sha256": manifest.get("source_sha256"),
            "chunking": manifest.get("chunking"),
        }

    return provenance


def _validate_cutoffs(
    cutoffs: Sequence[int],
) -> tuple[int, ...]:
    normalized_cutoffs = tuple(
        sorted(set(cutoffs))
    )

    if not normalized_cutoffs or any(
        isinstance(k, bool)
        or not isinstance(k, int)
        or k < 1
        for k in normalized_cutoffs
    ):
        raise ValueError(
            "cutoffs must contain positive integers"
        )

    return normalized_cutoffs


async def _run_search(
    operation: Awaitable[list[EvidenceChunk]],
) -> tuple[list[EvidenceChunk], float, str | None]:
    started_at = time.perf_counter()

    try:
        results = await operation
        error = None
    except Exception as exc:
        results = []
        error = f"{type(exc).__name__}: {exc}"

    duration_ms = (
        time.perf_counter() - started_at
    ) * 1000.0

    return results, duration_ms, error


def _calculate_ranking_metrics(
    retrieved_ids: list[str],
    relevance_by_chunk_id: dict[str, float],
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    relevant_ids = {
        chunk_id
        for chunk_id, grade in relevance_by_chunk_id.items()
        if grade > 0
    }

    return {
        "recall_at_k": {
            str(k): recall_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )
            for k in cutoffs
        },
        "precision_at_k": {
            str(k): precision_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )
            for k in cutoffs
        },
        "ndcg_at_k": {
            str(k): ndcg_at_k(
                retrieved_ids,
                relevance_by_chunk_id,
                k,
            )
            for k in cutoffs
        },
        "reciprocal_rank": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
    }


def _percentile_95(values: list[float]) -> float:
    ordered_values = sorted(values)
    index = max(
        0,
        math.ceil(0.95 * len(ordered_values)) - 1,
    )
    return ordered_values[index]


def _summarise_system(
    query_results: list[dict[str, Any]],
    system_name: str,
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    system_results = [
        result["systems"][system_name]
        for result in query_results
    ]
    query_count = len(system_results)

    return {
        "query_count": query_count,
        "failure_count": sum(
            result["error"] is not None
            for result in system_results
        ),
        "recall_at_k": {
            str(k): sum(
                result["metrics"]["recall_at_k"][str(k)]
                for result in system_results
            )
            / query_count
            for k in cutoffs
        },
        "precision_at_k": {
            str(k): sum(
                result["metrics"]["precision_at_k"][str(k)]
                for result in system_results
            )
            / query_count
            for k in cutoffs
        },
        "ndcg_at_k": {
            str(k): sum(
                result["metrics"]["ndcg_at_k"][str(k)]
                for result in system_results
            )
            / query_count
            for k in cutoffs
        },
        "mean_reciprocal_rank": sum(
            result["metrics"]["reciprocal_rank"]
            for result in system_results
        )
        / query_count,
        "mean_latency_ms": sum(
            result["latency_ms"]
            for result in system_results
        )
        / query_count,
        "p95_latency_ms": _percentile_95(
            [
                result["latency_ms"]
                for result in system_results
            ]
        ),
    }


async def run_benchmark(
    *,
    chunks: list[EvidenceChunk],
    queries: list[RetrievalBenchmarkQuery],
    embedding_client: EmbeddingClient | None = None,
    cutoffs: Sequence[int] = (4, 10),
    candidate_top_k: int | None = None,
    rrf_k: int = 60,
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("benchmark chunks must not be empty")

    if not queries:
        raise ValueError("benchmark queries must not be empty")

    normalized_cutoffs = _validate_cutoffs(cutoffs)
    evaluation_depth = max(normalized_cutoffs)

    if candidate_top_k is None:
        candidate_top_k = max(
            evaluation_depth * 2,
            10,
        )

    if candidate_top_k < evaluation_depth:
        raise ValueError(
            "candidate_top_k must be at least the largest cutoff"
        )

    validate_query_references(chunks, queries)

    chunks_by_paper: dict[str, list[EvidenceChunk]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_paper[chunk.paper_id].append(chunk)

    shared_embedding_client = (
        embedding_client or OllamaEmbeddingClient()
    )

    bm25_by_paper = {
        paper_id: BM25Retriever(paper_chunks)
        for paper_id, paper_chunks in chunks_by_paper.items()
    }
    vector_by_paper = {
        paper_id: VectorRetriever(
            paper_chunks,
            embedding_client=shared_embedding_client,
        )
        for paper_id, paper_chunks in chunks_by_paper.items()
    }
    hybrid_by_paper = {
        paper_id: HybridRetriever(
            bm25_by_paper[paper_id],
            vector_by_paper[paper_id],
            rrf_k=rrf_k,
        )
        for paper_id in chunks_by_paper
    }

    # Build document vectors before measuring query latency. Without this
    # warm-up, the first vector query would include indexing time while the
    # following hybrid query would reuse the cache, making latency numbers
    # incomparable. A failed warm-up is intentionally not fatal: individual
    # vector searches will record the failure and Hybrid can still use BM25.
    for vector_retriever in vector_by_paper.values():
        try:
            await vector_retriever.build_index()
        except Exception:
            continue

    query_results: list[dict[str, Any]] = []

    for query_case in queries:
        bm25 = bm25_by_paper[query_case.paper_id]
        vector = vector_by_paper[query_case.paper_id]
        hybrid = hybrid_by_paper[query_case.paper_id]

        operations: dict[
            str,
            Callable[[], Awaitable[list[EvidenceChunk]]],
        ] = {
            "bm25": lambda: asyncio.to_thread(
                bm25.search,
                query_case.query,
                top_k=evaluation_depth,
            ),
            "vector": lambda: vector.search(
                query_case.query,
                top_k=evaluation_depth,
            ),
            "hybrid": lambda: hybrid.search(
                query_case.query,
                top_k=evaluation_depth,
                candidate_top_k=candidate_top_k,
            ),
        }

        system_results: dict[str, Any] = {}

        for system_name in SYSTEM_NAMES:
            results, latency_ms, error = await _run_search(
                operations[system_name]()
            )
            retrieved_ids = [
                result.chunk_id
                for result in results
            ]

            system_results[system_name] = {
                "retrieved_chunk_ids": retrieved_ids,
                "latency_ms": latency_ms,
                "error": error,
                "metrics": _calculate_ranking_metrics(
                    retrieved_ids,
                    query_case.relevance_by_chunk_id,
                    normalized_cutoffs,
                ),
            }

        query_results.append(
            {
                "query_id": query_case.query_id,
                "paper_id": query_case.paper_id,
                "dimension": query_case.dimension,
                "query": query_case.query,
                "systems": system_results,
            }
        )

    summary = {
        system_name: _summarise_system(
            query_results,
            system_name,
            normalized_cutoffs,
        )
        for system_name in SYSTEM_NAMES
    }

    return {
        "config": {
            "cutoffs": list(normalized_cutoffs),
            "candidate_top_k": candidate_top_k,
            "rrf_k": rrf_k,
        },
        "summary": summary,
        "queries": query_results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BM25, vector and hybrid academic evidence retrieval."
        )
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[4, 10],
    )
    parser.add_argument("--candidate-top-k", type=int)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--embedding-model")
    parser.add_argument(
        "--label-provenance",
        help="Optional JSON metadata describing who or what assigned qrels.",
    )
    parser.add_argument(
        "--dataset-manifest",
        help="Optional frozen-dataset manifest JSON.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    corpus_path = Path(args.corpus).expanduser().resolve()
    queries_path = Path(args.queries).expanduser().resolve()
    label_provenance_path = (
        Path(args.label_provenance).expanduser().resolve()
        if args.label_provenance
        else None
    )
    dataset_manifest_path = (
        Path(args.dataset_manifest).expanduser().resolve()
        if args.dataset_manifest
        else None
    )

    chunks = load_corpus_jsonl(corpus_path)
    queries = load_queries_jsonl(queries_path)
    embedding_client = OllamaEmbeddingClient(
        model=args.embedding_model,
    )

    report = await run_benchmark(
        chunks=chunks,
        queries=queries,
        embedding_client=embedding_client,
        cutoffs=args.cutoffs,
        candidate_top_k=args.candidate_top_k,
        rrf_k=args.rrf_k,
    )
    report["provenance"] = _build_provenance(
        corpus_path=corpus_path,
        queries_path=queries_path,
        embedding_model=embedding_client.model,
        label_provenance_path=label_provenance_path,
        dataset_manifest_path=dataset_manifest_path,
    )

    rendered_report = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            rendered_report + "\n",
            encoding="utf-8",
        )
        print(f"Benchmark report written to {output_path}")
    else:
        print(rendered_report)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
