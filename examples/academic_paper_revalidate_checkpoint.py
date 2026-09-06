from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from gpt_researcher.academic.citation_validator import (
    CitationValidator,
    CitationValidatorError,
)
from gpt_researcher.academic.evaluation_citation_verifier import (
    EvaluationCitationVerifierError,
    build_claim_support_results,
    verify_evaluation_claims,
)
from gpt_researcher.academic.ollama_citation_client import (
    OllamaCitationClient,
)
from gpt_researcher.academic.paper_chunker import (
    AcademicPaperChunker,
)
from gpt_researcher.academic.paper_evaluation_pipeline import (
    PaperEvaluationPipeline,
)
from gpt_researcher.academic.paper_evaluator import (
    PaperEvaluator,
)
from gpt_researcher.academic.schemas import (
    DimensionEvaluation,
    EvidenceChunk,
)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Checkpoint must contain a JSON object."
        )

    return payload


def write_checkpoint(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def needs_revalidation(
    result: DimensionEvaluation,
) -> bool:
    if result.verification is None:
        return True

    support_rate = (
        result.verification.metrics.claim_support_rate
    )

    return (
        support_rate is None
        or support_rate == 0.0
    )


def save_revalidation_state(
    *,
    checkpoint_path: Path,
    payload: dict[str, Any],
    paper_id: str,
    chunks: list[EvidenceChunk],
    results: list[DimensionEvaluation],
    revalidated_dimensions: set[str],
    errors: dict[str, str],
) -> None:
    pending_dimensions = [
        result.dimension
        for result in results
        if (
            needs_revalidation(result)
            and result.dimension
            not in revalidated_dimensions
        )
    ]

    final_evaluation = PaperEvaluator().evaluate(
        paper_id=paper_id,
        evidence_chunks=chunks,
        dimension_evaluations=results,
    )

    payload["dimension_results"] = [
        result.model_dump(mode="json")
        for result in results
    ]
    payload["final_evaluation"] = (
        final_evaluation.model_dump(mode="json")
    )
    payload["citation_revalidated_dimensions"] = sorted(
        revalidated_dimensions
    )
    payload["citation_revalidation_errors"] = errors
    payload["citation_revalidation_pending_dimensions"] = (
        pending_dimensions
    )
    payload["citation_revalidation_completed"] = (
        not pending_dimensions
    )

    write_checkpoint(
        checkpoint_path,
        payload,
    )


async def run_revalidation(
    args: argparse.Namespace,
) -> None:
    checkpoint_path = Path(
        args.checkpoint
    ).expanduser().resolve()

    payload = load_checkpoint(checkpoint_path)

    paper_id = payload.get("paper_id")
    if not isinstance(paper_id, str) or not paper_id:
        raise ValueError(
            "Checkpoint does not contain a valid paper_id."
        )

    raw_results = payload.get(
        "dimension_results",
        [],
    )
    if not isinstance(raw_results, list):
        raise ValueError(
            "dimension_results must be a list."
        )

    results = [
        DimensionEvaluation.model_validate(item)
        for item in raw_results
    ]

    revalidated_dimensions = set(
        payload.get(
            "citation_revalidated_dimensions",
            [],
        )
    )

    targets = [
        result.dimension
        for result in results
        if (
            needs_revalidation(result)
            and result.dimension
            not in revalidated_dimensions
        )
    ]

    print(f"Paper ID: {paper_id}")
    print(
        "Dimensions requiring citation revalidation: "
        f"{len(targets)}"
    )

    for dimension in targets:
        print(f"- {dimension}")

    if args.dry_run:
        print("\nDry run only. No files were changed.")
        return

    if not targets:
        print("No citation revalidation is required.")
        return

    pdf_path = Path(
        args.pdf
    ).expanduser().resolve()

    print("\nLoading and chunking PDF...")

    chunker = AcademicPaperChunker(
        target_tokens=800,
        overlap_tokens=120,
        min_chunk_tokens=100,
    )
    chunks = chunker.chunk_pdf(
        file_path=pdf_path,
        paper_id=paper_id,
        source_url=args.source_url,
    )

    chunk_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    citation_validator = CitationValidator(
        OllamaCitationClient()
    )

    errors = dict(
        payload.get(
            "citation_revalidation_errors",
            {},
        )
    )

    for index, result in enumerate(results):
        dimension = result.dimension

        if dimension not in targets:
            continue

        missing_chunk_ids = [
            chunk_id
            for chunk_id in result.evidence_chunk_ids
            if chunk_id not in chunk_by_id
        ]

        selected_chunks = [
            chunk_by_id[chunk_id]
            for chunk_id in result.evidence_chunk_ids
            if chunk_id in chunk_by_id
        ]

        if missing_chunk_ids:
            message = (
                "Checkpoint references missing chunks: "
                + ", ".join(missing_chunk_ids)
            )
            errors[dimension] = message
            print(f"\n{dimension}: {message}")

            save_revalidation_state(
                checkpoint_path=checkpoint_path,
                payload=payload,
                paper_id=paper_id,
                chunks=chunks,
                results=results,
                revalidated_dimensions=(
                    revalidated_dimensions
                ),
                errors=errors,
            )
            continue

        print(
            f"\nRevalidating {dimension} "
            f"with {len(selected_chunks)} chunks..."
        )

        try:
            validations = await verify_evaluation_claims(
                validator=citation_validator,
                paper_id=paper_id,
                evaluation=result,
                evidence_chunks=selected_chunks,
                max_claims=args.max_claims,
            )
            claim_support_results = (
                build_claim_support_results(
                    validations
                )
            )
        except (
            CitationValidatorError,
            EvaluationCitationVerifierError,
        ) as exc:
            errors[dimension] = str(exc)

            results[index] = (
                PaperEvaluationPipeline
                ._attach_verification(
                    result=result,
                    selected_chunks=selected_chunks,
                    claim_support_results=None,
                )
            )

            print(
                f"{dimension}: validation failed: {exc}"
            )

            save_revalidation_state(
                checkpoint_path=checkpoint_path,
                payload=payload,
                paper_id=paper_id,
                chunks=chunks,
                results=results,
                revalidated_dimensions=(
                    revalidated_dimensions
                ),
                errors=errors,
            )
            continue

        results[index] = (
            PaperEvaluationPipeline._attach_verification(
                result=result,
                selected_chunks=selected_chunks,
                claim_support_results=(
                    claim_support_results
                ),
            )
        )

        revalidated_dimensions.add(dimension)
        errors.pop(dimension, None)

        save_revalidation_state(
            checkpoint_path=checkpoint_path,
            payload=payload,
            paper_id=paper_id,
            chunks=chunks,
            results=results,
            revalidated_dimensions=(
                revalidated_dimensions
            ),
            errors=errors,
        )

        print(
            f"{dimension}: citation verification saved."
        )

    save_revalidation_state(
        checkpoint_path=checkpoint_path,
        payload=payload,
        paper_id=paper_id,
        chunks=chunks,
        results=results,
        revalidated_dimensions=revalidated_dimensions,
        errors=errors,
    )

    pending = payload.get(
        "citation_revalidation_pending_dimensions",
        [],
    )

    print("\nCitation revalidation finished.")
    print(f"Pending dimensions: {len(pending)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate citations from a completed "
            "academic evaluation checkpoint."
        )
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the original academic PDF.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the completed checkpoint JSON.",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional source URL.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=1,
        help="Claims validated per dimension.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List target dimensions without changing "
            "the checkpoint."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(
        run_revalidation(parse_args())
    )