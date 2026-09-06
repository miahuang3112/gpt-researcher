from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpt_researcher.academic.vector_retriever import (
    EmbeddingClient,
    OllamaEmbeddingClient,
)

from .dataset import load_corpus_jsonl, load_queries_jsonl
from .run_benchmark import run_benchmark
from .split_collection import assert_test_manifest
from .tune_retrieval import assert_complete_trial


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load {description} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def validate_frozen_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("status") != "frozen":
        raise ValueError("Held-out evaluation requires a frozen retrieval policy")
    if policy.get("selection_scope") != "development-only":
        raise ValueError("Frozen policy must have been selected on development only")
    if policy.get("retriever") != "hybrid":
        raise ValueError("Only the frozen Hybrid policy is supported")
    integrity = policy.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("retuning_after_test_permitted") is not False:
        raise ValueError("Frozen policy must explicitly forbid test-driven retuning")
    parameters = policy.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Frozen policy has no parameters")
    required = ("embedding_model", "candidate_top_k", "rrf_k", "evaluation_cutoffs")
    if any(key not in parameters for key in required):
        raise ValueError("Frozen policy parameters are incomplete")
    return parameters


async def evaluate_held_out_test(
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
    client = embedding_client or OllamaEmbeddingClient(
        model=parameters["embedding_model"],
        timeout=600.0,
        batch_size=8,
    )
    benchmark = await run_benchmark(
        chunks=chunks,
        queries=queries,
        embedding_client=client,
        cutoffs=parameters["evaluation_cutoffs"],
        candidate_top_k=parameters["candidate_top_k"],
        rrf_k=parameters["rrf_k"],
    )
    assert_complete_trial(benchmark)
    return {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "held-out-test",
        "parameter_selection_permitted": False,
        "policy": {
            "path": str(resolved_policy),
            "sha256": _sha256(resolved_policy),
            "policy_id": policy["policy_id"],
            "retriever": policy["retriever"],
            "parameters": parameters,
        },
        "test_split": {
            "manifest": str(resolved_manifest),
            "manifest_sha256": _sha256(resolved_manifest),
            "paper_ids": test_manifest["paper_ids"],
            "corpus_sha256": test_manifest["corpus_sha256"],
            "queries_sha256": test_manifest["queries_sha256"],
        },
        "default_system": "hybrid",
        "default_system_summary": benchmark["summary"]["hybrid"],
        "benchmark": benchmark,
        "integrity_warning": (
            "This is a one-time held-out report. Test results must not be used "
            "to change this frozen policy; any later policy requires a new "
            "development split and policy version."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an immutable retrieval policy once on held-out test data."
    )
    parser.add_argument("--policy", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    report = await evaluate_held_out_test(
        policy_path=args.policy,
        test_manifest_path=args.test_manifest,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Held-out report written to {output_path}")
    print(f"Frozen default summary: {report['default_system_summary']}")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
