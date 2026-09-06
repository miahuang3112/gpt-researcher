from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.academic_quality.measure_score_reliability import (
    _safe_error,
    load_frozen_selections,
)
from evals.academic_retrieval.dataset import load_corpus_jsonl
from gpt_researcher.academic.ollama_paper_evaluator_client import (
    OllamaPaperEvaluatorClient,
)
from gpt_researcher.academic.paper_evaluation_pipeline import (
    PaperEvaluationPipeline,
)


def _safe_json_shape(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "json_valid": False,
            "json_error": exc.msg,
            "json_error_position": exc.pos,
        }
    if not isinstance(value, dict):
        return {"json_valid": True, "root_type": type(value).__name__}
    return {
        "json_valid": True,
        "root_type": "object",
        "keys": sorted(value),
        "dimension": value.get("dimension"),
        "status": value.get("status"),
        "score_type": type(value.get("score")).__name__,
        "evidence_chunk_ids": value.get("evidence_chunk_ids"),
        "external_evidence_ids": value.get("external_evidence_ids"),
    }


async def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    development_report = Path(args.development_report).resolve()
    heldout_report = Path(args.heldout_report).resolve()
    selections, _ = load_frozen_selections(development_report, heldout_report)
    key = (args.paper_id, args.dimension)
    if key not in selections:
        raise ValueError(f"No frozen selection for {key}.")
    chunks = []
    for manifest_path in (args.development_manifest, args.heldout_manifest):
        manifest = Path(manifest_path).resolve()
        chunks.extend(load_corpus_jsonl(manifest.parent / "corpus.jsonl"))
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    selected_ids = selections[key]
    prompt = PaperEvaluationPipeline.build_prompt(
        args.paper_id,
        args.dimension,
        [chunks_by_id[chunk_id] for chunk_id in selected_ids],
    )
    client = OllamaPaperEvaluatorClient(
        model=args.model,
        base_url=args.base_url,
        timeout=600.0,
        temperature=0.0,
        think=False,
    )
    try:
        raw = await client.generate(prompt)
    except Exception as exc:
        diagnostic = {
            "paper_id": args.paper_id,
            "dimension": args.dimension,
            "model": args.model,
            "selected_chunk_ids": selected_ids,
            "generation_ok": False,
            "error": _safe_error(exc),
        }
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return diagnostic
    diagnostic = {
        "paper_id": args.paper_id,
        "dimension": args.dimension,
        "model": args.model,
        "selected_chunk_ids": selected_ids,
        "generation_ok": True,
        "raw_length": len(raw),
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        **_safe_json_shape(raw),
    }
    try:
        PaperEvaluationPipeline.parse_dimension_response(
            raw,
            expected_dimension=args.dimension,
            allowed_chunk_ids=set(selected_ids),
        )
    except Exception as exc:
        diagnostic["parse_ok"] = False
        diagnostic["error_type"] = type(exc).__name__
        diagnostic["safe_error"] = str(exc)[:1000]
    else:
        diagnostic["parse_ok"] = True
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--development-report", default="evals/academic_retrieval/results/ret09-development-tuning-qwen3-embedding-4b.json")
    parser.add_argument("--heldout-report", default="evals/academic_retrieval/results/ret10b-heldout-test-qwen3-embedding-4b.json")
    parser.add_argument("--development-manifest", default="evals/academic_retrieval/datasets/ret09-five-paper-v1/development/manifest.json")
    parser.add_argument("--heldout-manifest", default="evals/academic_retrieval/datasets/ret09-five-paper-v1/test/manifest.json")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(diagnose(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
