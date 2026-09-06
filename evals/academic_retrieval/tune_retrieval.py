from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpt_researcher.academic.vector_retriever import EmbeddingClient, OllamaEmbeddingClient

from .dataset import load_corpus_jsonl, load_queries_jsonl
from .run_benchmark import run_benchmark
from .split_collection import assert_tuning_manifest


class CachedEmbeddingClient:
    """Reuse document and query embeddings across a development-only grid search."""

    def __init__(self, client: EmbeddingClient) -> None:
        self.client = client
        self.cache: dict[str, list[float]] = {}

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ordered = list(texts)
        missing = list(dict.fromkeys(text for text in ordered if text not in self.cache))
        if missing:
            vectors = await self.client.embed(missing)
            if len(vectors) != len(missing):
                raise RuntimeError("Embedding client returned an unexpected vector count")
            self.cache.update(zip(missing, vectors, strict=True))
        return [self.cache[text] for text in ordered]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_value(trial: dict[str, Any], metric: str, cutoff: int) -> float:
    return float(trial["summary"]["hybrid"][metric][str(cutoff)])


def select_best_trial(
    trials: list[dict[str, Any]],
    *,
    primary_metric: str = "recall_at_k",
    cutoff: int = 10,
) -> dict[str, Any]:
    if not trials:
        raise ValueError("At least one tuning trial is required")
    if primary_metric not in {"recall_at_k", "ndcg_at_k"}:
        raise ValueError("primary_metric must be recall_at_k or ndcg_at_k")
    secondary_metric = "ndcg_at_k" if primary_metric == "recall_at_k" else "recall_at_k"
    return max(
        trials,
        key=lambda trial: (
            _metric_value(trial, primary_metric, cutoff),
            _metric_value(trial, secondary_metric, cutoff),
            -trial["config"]["candidate_top_k"],
            -trial["config"]["rrf_k"],
        ),
    )


def assert_complete_trial(trial: dict[str, Any]) -> None:
    failures = {
        system_name: int(system_summary["failure_count"])
        for system_name, system_summary in trial["summary"].items()
        if int(system_summary["failure_count"]) > 0
    }
    if failures:
        raise RuntimeError(
            "Refusing to tune from an incomplete retrieval trial; "
            f"failures={failures}"
        )


async def tune_on_development(
    *,
    development_manifest_path: str | Path,
    embedding_client: EmbeddingClient,
    candidate_top_k_values: Sequence[int] = (10, 20, 40),
    rrf_k_values: Sequence[int] = (20, 60, 100),
    cutoffs: Sequence[int] = (4, 10),
    primary_metric: str = "recall_at_k",
) -> dict[str, Any]:
    manifest_path = Path(development_manifest_path).expanduser().resolve()
    manifest = assert_tuning_manifest(manifest_path)
    chunks = load_corpus_jsonl(manifest_path.parent / "corpus.jsonl")
    queries = load_queries_jsonl(manifest_path.parent / "queries.jsonl")
    normalized_candidates = tuple(sorted(set(candidate_top_k_values)))
    normalized_rrf = tuple(sorted(set(rrf_k_values)))
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not normalized_candidates or not normalized_rrf:
        raise ValueError("Tuning grids must not be empty")
    if any(value < max(normalized_cutoffs) for value in normalized_candidates):
        raise ValueError("candidate_top_k values must cover the largest cutoff")
    if any(value < 1 for value in normalized_rrf):
        raise ValueError("rrf_k values must be positive")

    cached_client = CachedEmbeddingClient(embedding_client)
    trials: list[dict[str, Any]] = []
    for candidate_top_k in normalized_candidates:
        for rrf_k in normalized_rrf:
            trial = await run_benchmark(
                chunks=chunks,
                queries=queries,
                embedding_client=cached_client,
                cutoffs=normalized_cutoffs,
                candidate_top_k=candidate_top_k,
                rrf_k=rrf_k,
            )
            assert_complete_trial(trial)
            trials.append(trial)

    objective_cutoff = max(normalized_cutoffs)
    best = select_best_trial(
        trials,
        primary_metric=primary_metric,
        cutoff=objective_cutoff,
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "development-only",
        "test_split_accessed": False,
        "development": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "corpus_sha256": manifest["corpus_sha256"],
            "queries_sha256": manifest["queries_sha256"],
            "paper_ids": manifest["paper_ids"],
        },
        "objective": {
            "system": "hybrid",
            "primary_metric": primary_metric,
            "secondary_metric": (
                "ndcg_at_k" if primary_metric == "recall_at_k" else "recall_at_k"
            ),
            "cutoff": objective_cutoff,
            "tie_breaker": "smaller candidate_top_k, then smaller rrf_k",
        },
        "search_space": {
            "candidate_top_k": list(normalized_candidates),
            "rrf_k": list(normalized_rrf),
        },
        "selected_config": best["config"],
        "selected_development_metrics": best["summary"]["hybrid"],
        "trials": trials,
        "usage_warning": (
            "This configuration was selected only on development papers. "
            "Run it once on the held-out test split for final reporting; never retune from test results."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hybrid retrieval on a frozen development split.")
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--candidate-top-k", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--rrf-k", type=int, nargs="+", default=[20, 60, 100])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--primary-metric", choices=["recall_at_k", "ndcg_at_k"], default="recall_at_k")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-timeout", type=float, default=600.0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    client = OllamaEmbeddingClient(
        model=args.embedding_model,
        timeout=args.embedding_timeout,
        batch_size=args.embedding_batch_size,
    )
    report = await tune_on_development(
        development_manifest_path=args.development_manifest,
        embedding_client=client,
        candidate_top_k_values=args.candidate_top_k,
        rrf_k_values=args.rrf_k,
        cutoffs=args.cutoffs,
        primary_metric=args.primary_metric,
    )
    report["embedding"] = {"provider": "ollama", "model": client.model}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Development tuning report written to {output}")
    print(f"Selected config: {report['selected_config']}")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
