from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpt_researcher.academic.bm25_retriever import BM25Retriever
from gpt_researcher.academic.hybrid_retriever import HybridRetriever
from gpt_researcher.academic.paper_evaluation_pipeline import PaperEvaluationPipeline
from gpt_researcher.academic.schemas import EvidenceChunk
from gpt_researcher.academic.vector_retriever import (
    EmbeddingClient,
    OllamaEmbeddingClient,
    VectorRetriever,
)

from .dataset import RetrievalBenchmarkQuery, load_corpus_jsonl, load_queries_jsonl
from .evaluate_frozen_policy import _load_object, validate_frozen_policy
from .metrics import ndcg_at_k, precision_at_k, recall_at_k
from .split_collection import assert_test_manifest


class _UnusedEvaluationClient:
    async def generate(self, prompt: str) -> str:
        raise AssertionError("Selector comparison must not call a scoring model")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _automatic_metrics(
    selected: Sequence[EvidenceChunk],
    query: RetrievalBenchmarkQuery,
    *,
    top_k: int,
    corpus_chunk_ids: set[str],
) -> dict[str, Any]:
    selected_ids = [chunk.chunk_id for chunk in selected]
    relevance = query.relevance_by_chunk_id
    relevant_ids = {chunk_id for chunk_id, grade in relevance.items() if grade > 0}
    highly_relevant_ids = {
        chunk_id for chunk_id, grade in relevance.items() if grade >= 2
    }
    valid_count = sum(chunk_id in corpus_chunk_ids for chunk_id in selected_ids)
    denominator = len(selected_ids)
    return {
        "evidence": {
            "recall_at_k": recall_at_k(selected_ids, relevant_ids, top_k),
            "precision_at_k": precision_at_k(selected_ids, relevant_ids, top_k),
            "ndcg_at_k": ndcg_at_k(selected_ids, relevance, top_k),
        },
        "citation_readiness_proxy": {
            "valid_candidate_rate": valid_count / denominator if denominator else 0.0,
            "relevant_candidate_rate": (
                sum(chunk_id in relevant_ids for chunk_id in selected_ids) / denominator
                if denominator
                else 0.0
            ),
            "highly_relevant_candidate_rate": (
                sum(chunk_id in highly_relevant_ids for chunk_id in selected_ids)
                / denominator
                if denominator
                else 0.0
            ),
            "query_has_relevant_candidate": any(
                chunk_id in relevant_ids for chunk_id in selected_ids
            ),
        },
    }


def _summarize(query_records: Sequence[dict[str, Any]], system_name: str) -> dict[str, Any]:
    records = [record["systems"][system_name] for record in query_records]
    successes = [record for record in records if record["error"] is None]
    count = len(records)

    def mean(path: tuple[str, str]) -> float:
        if not successes:
            return 0.0
        return sum(record[path[0]][path[1]] for record in successes) / len(successes)

    latencies = [record["latency_ms"] for record in successes]
    return {
        "query_count": count,
        "failure_count": count - len(successes),
        "evidence": {
            "recall_at_k": mean(("evidence", "recall_at_k")),
            "precision_at_k": mean(("evidence", "precision_at_k")),
            "ndcg_at_k": mean(("evidence", "ndcg_at_k")),
        },
        "citation_readiness_proxy": {
            "valid_candidate_rate": mean(
                ("citation_readiness_proxy", "valid_candidate_rate")
            ),
            "relevant_candidate_rate": mean(
                ("citation_readiness_proxy", "relevant_candidate_rate")
            ),
            "highly_relevant_candidate_rate": mean(
                ("citation_readiness_proxy", "highly_relevant_candidate_rate")
            ),
            "query_coverage_rate": (
                sum(
                    record["citation_readiness_proxy"][
                        "query_has_relevant_candidate"
                    ]
                    for record in successes
                )
                / len(successes)
                if successes
                else 0.0
            ),
        },
        "latency_ms": {
            "mean_query": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95_query": _p95(latencies),
        },
    }


async def compare_selectors(
    *,
    policy_path: str | Path,
    test_manifest_path: str | Path,
    embedding_client: EmbeddingClient | None = None,
) -> dict[str, Any]:
    resolved_policy = Path(policy_path).expanduser().resolve()
    resolved_manifest = Path(test_manifest_path).expanduser().resolve()
    policy = _load_object(resolved_policy, "frozen retrieval policy")
    parameters = validate_frozen_policy(policy)
    test_manifest = assert_test_manifest(resolved_manifest)
    chunks = load_corpus_jsonl(resolved_manifest.parent / "corpus.jsonl")
    queries = load_queries_jsonl(resolved_manifest.parent / "queries.jsonl")
    top_k = int(parameters["evidence_top_k_default"])
    candidate_top_k = int(parameters["candidate_top_k"])
    rrf_k = int(parameters["rrf_k"])
    corpus_chunk_ids = {chunk.chunk_id for chunk in chunks}

    chunks_by_paper: dict[str, list[EvidenceChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_paper[chunk.paper_id].append(chunk)

    client = embedding_client or OllamaEmbeddingClient(
        model=parameters["embedding_model"],
        timeout=600.0,
        batch_size=8,
    )
    section_selector = PaperEvaluationPipeline(
        client=_UnusedEvaluationClient(),
        max_chunks_per_dimension=top_k,
    )
    hybrid_by_paper: dict[str, HybridRetriever] = {}
    index_build_records: list[dict[str, Any]] = []
    for paper_id, paper_chunks in chunks_by_paper.items():
        vector = VectorRetriever(paper_chunks, embedding_client=client)
        started = time.perf_counter()
        error: str | None = None
        try:
            await vector.build_index()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        hybrid_by_paper[paper_id] = HybridRetriever(
            BM25Retriever(paper_chunks),
            vector,
            rrf_k=rrf_k,
        )
        index_build_records.append(
            {"paper_id": paper_id, "latency_ms": elapsed_ms, "error": error}
        )

    query_records: list[dict[str, Any]] = []
    for query in queries:
        systems: dict[str, Any] = {}

        section_started = time.perf_counter()
        try:
            section_results = section_selector.select_evidence(
                query.dimension,
                chunks_by_paper[query.paper_id],
            )
            section_error = None
        except Exception as exc:
            section_results = []
            section_error = f"{type(exc).__name__}: {exc}"
        section_latency = (time.perf_counter() - section_started) * 1000.0
        systems["section"] = {
            "selected_chunk_ids": [chunk.chunk_id for chunk in section_results],
            "latency_ms": section_latency,
            "error": section_error,
            **_automatic_metrics(
                section_results,
                query,
                top_k=top_k,
                corpus_chunk_ids=corpus_chunk_ids,
            ),
        }

        hybrid_started = time.perf_counter()
        try:
            hybrid_results = await hybrid_by_paper[query.paper_id].search(
                query.query,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
            )
            hybrid_error = None
        except Exception as exc:
            hybrid_results = []
            hybrid_error = f"{type(exc).__name__}: {exc}"
        hybrid_latency = (time.perf_counter() - hybrid_started) * 1000.0
        systems["hybrid"] = {
            "selected_chunk_ids": [chunk.chunk_id for chunk in hybrid_results],
            "latency_ms": hybrid_latency,
            "error": hybrid_error,
            **_automatic_metrics(
                hybrid_results,
                query,
                top_k=top_k,
                corpus_chunk_ids=corpus_chunk_ids,
            ),
        }
        query_records.append(
            {
                "query_id": query.query_id,
                "paper_id": query.paper_id,
                "dimension": query.dimension,
                "systems": systems,
            }
        )

    summaries = {
        name: _summarize(query_records, name) for name in ("section", "hybrid")
    }
    index_failures = sum(record["error"] is not None for record in index_build_records)
    summaries["hybrid"]["index_build"] = {
        "paper_count": len(index_build_records),
        "failure_count": index_failures,
        "total_latency_ms": sum(record["latency_ms"] for record in index_build_records),
        "mean_latency_ms": (
            sum(record["latency_ms"] for record in index_build_records)
            / len(index_build_records)
            if index_build_records
            else 0.0
        ),
        "records": index_build_records,
    }

    delta = {
        "evidence_recall_at_k": (
            summaries["hybrid"]["evidence"]["recall_at_k"]
            - summaries["section"]["evidence"]["recall_at_k"]
        ),
        "evidence_ndcg_at_k": (
            summaries["hybrid"]["evidence"]["ndcg_at_k"]
            - summaries["section"]["evidence"]["ndcg_at_k"]
        ),
        "relevant_candidate_rate": (
            summaries["hybrid"]["citation_readiness_proxy"][
                "relevant_candidate_rate"
            ]
            - summaries["section"]["citation_readiness_proxy"][
                "relevant_candidate_rate"
            ]
        ),
        "mean_query_latency_ms": (
            summaries["hybrid"]["latency_ms"]["mean_query"]
            - summaries["section"]["latency_ms"]["mean_query"]
        ),
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": "held-out-test-descriptive-only",
        "policy_changed": False,
        "policy": {
            "path": str(resolved_policy),
            "sha256": _sha256(resolved_policy),
            "policy_id": policy["policy_id"],
            "parameters": parameters,
        },
        "test_split": {
            "manifest": str(resolved_manifest),
            "manifest_sha256": _sha256(resolved_manifest),
            "paper_ids": test_manifest["paper_ids"],
            "corpus_sha256": test_manifest["corpus_sha256"],
            "queries_sha256": test_manifest["queries_sha256"],
        },
        "config": {"top_k": top_k, "candidate_top_k": candidate_top_k, "rrf_k": rrf_k},
        "summary": summaries,
        "hybrid_minus_section": delta,
        "queries": query_records,
        "interpretation_boundary": {
            "citation_metric_type": "qrels-based citation-readiness proxy",
            "measures_actual_model_citation_choice": False,
            "human_paper_quality_measured": False,
            "retuning_from_this_report_permitted": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare section-only and frozen Hybrid evidence selection."
    )
    parser.add_argument("--policy", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    report = await compare_selectors(
        policy_path=args.policy,
        test_manifest_path=args.test_manifest,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selector comparison written to {output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
