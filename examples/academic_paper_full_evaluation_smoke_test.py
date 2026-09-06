from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from gpt_researcher.academic.ollama_paper_evaluator_client import (
    OllamaPaperEvaluatorClient,
)
from gpt_researcher.academic.paper_chunker import (
    AcademicPaperChunker,
)
from gpt_researcher.academic.paper_evaluation_pipeline import (
    PaperEvaluationPipeline,
)

from gpt_researcher.academic.citation_validator import (
    CitationValidator,
)

from gpt_researcher.academic.ollama_citation_client import (
    OllamaCitationClient,
)

from gpt_researcher.academic.schemas import (
    DimensionEvaluation,
)

def write_checkpoint(
    checkpoint_path: Path,
    payload: dict,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    paper_id: str,
) -> list[DimensionEvaluation]:
    if not checkpoint_path.exists():
        return []

    payload = json.loads(
        checkpoint_path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("paper_id") != paper_id:
        raise ValueError(
            "Checkpoint paper_id does not match "
            f"the current paper: {checkpoint_path}"
        )

    raw_results = payload.get(
        "dimension_results",
        [],
    )

    if not isinstance(raw_results, list):
        raise ValueError(
            "Checkpoint dimension_results "
            "must be a list."
        )

    return [
        DimensionEvaluation.model_validate(item)
        for item in raw_results
    ]

def initialize_checkpoint(
    checkpoint_path: Path,
    paper_id: str,
) -> list[DimensionEvaluation]:
    completed_results = load_checkpoint(
        checkpoint_path,
        paper_id,
    )

    if not checkpoint_path.exists():
        write_checkpoint(
            checkpoint_path,
            {
                "paper_id": paper_id,
                "completed": False,
                "dimension_results": [],
            },
        )

        print(
            "Checkpoint initialized: "
            f"{checkpoint_path}"
        )

    return completed_results

async def run_smoke_test(args) -> None:
    started_at = time.perf_counter()
    pdf_path = Path(
        args.pdf
    ).expanduser().resolve()

    paper_id = (
        args.paper_id
        or pdf_path.stem
    )

    checkpoint_path = (
        Path(args.checkpoint)
        .expanduser()
        .resolve()
        if args.checkpoint
        else pdf_path.with_suffix(
            ".academic-evaluation-checkpoint.json"
        )
    )

    completed_dimension_results = (
        initialize_checkpoint(
            checkpoint_path,
            paper_id,
        )
    )

    latest_dimension_results = list(
        completed_dimension_results
    )

    if completed_dimension_results:
        print(
            "Resuming from checkpoint: "
            f"{len(completed_dimension_results)} "
            "dimensions already completed."
        )
    else:
        print(
            "Starting without completed "
            "checkpoint dimensions."
        )

    print(
        f"Checkpoint path: {checkpoint_path}"
    )

    print("Loading and chunking PDF...")

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

    print(f"Total PDF chunks: {len(chunks)}")


    evaluation_client = OllamaPaperEvaluatorClient(
        model=args.model,
    )

    citation_client = OllamaCitationClient()
    citation_validator = CitationValidator(
        citation_client
    )

    pipeline = PaperEvaluationPipeline(
        client=evaluation_client,
        max_chunks_per_dimension=args.max_chunks,
        citation_validator=citation_validator,
        max_claims_per_dimension=1,
    )

    def save_dimension_checkpoint(
        results,
    ) -> None:
        nonlocal latest_dimension_results

        latest_dimension_results = list(
            results
        )

        write_checkpoint(
            checkpoint_path,
            {
                "paper_id": paper_id,
                "completed": False,
                "dimension_results": [
                    item.model_dump(
                        mode="json"
                    )
                    for item in results
                ],
            },
        )

        print(
            "Checkpoint saved: "
            f"{len(results)} dimensions completed."
        )

    print("\nRunning full 9-dimension evaluation...")
    print(f"Paper ID: {paper_id}")
    print(f"Evaluation model: {args.model}")
    print(
        "Maximum evidence chunks per dimension: "
        f"{args.max_chunks}"
    )
    print("Claims validated per dimension: 1")

    result = await pipeline.evaluate(
        paper_id=paper_id,
        evidence_chunks=chunks,
        completed_dimension_results=(
            completed_dimension_results
        ),
        on_dimension_complete=(
            save_dimension_checkpoint
        ),
    )

    write_checkpoint(
        checkpoint_path,
        {
            "paper_id": paper_id,
            "completed": True,
            "dimension_results": [
                item.model_dump(mode="json")
                for item in latest_dimension_results
            ],
            "final_evaluation": (
                result.model_dump(mode="json")
            ),
        },
    )

    print(
        "Checkpoint marked as completed."
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    print("\nFULL PAPER EVALUATION")

    print(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nElapsed seconds: "
        f"{elapsed_seconds:.2f}"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one real Ollama full academic "
            "paper evaluation."
        )
    )

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to a local academic PDF.",
    )

    parser.add_argument(
        "--paper-id",
        default=None,
        help=(
            "Paper ID. Defaults to the PDF filename."
        ),
    )

    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional paper URL.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Ollama model, for example qwen3.5:9b."
        ),
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=4,
        help=(
            "Maximum Evidence Chunks supplied "
            "to the model."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint JSON path. Defaults "
            "to a file beside the PDF."
        ),
    )


    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(
        run_smoke_test(
            parse_args()
        )
    )