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
from gpt_researcher.academic.paper_evaluator import (
    REQUIRED_DIMENSIONS,
)

from gpt_researcher.academic.citation_validator import (
    CitationValidator,
)
from gpt_researcher.academic.evaluation_citation_verifier import (
    build_claim_support_results,
    verify_evaluation_claims,
)
from gpt_researcher.academic.evaluation_claims import (
    build_evaluation_claims,
)
from gpt_researcher.academic.ollama_citation_client import (
    OllamaCitationClient,
)


async def run_smoke_test(args) -> None:
    pdf_path = Path(
        args.pdf
    ).expanduser().resolve()

    paper_id = (
        args.paper_id
        or pdf_path.stem
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

    client = OllamaPaperEvaluatorClient(
        model=args.model,
    )

    pipeline = PaperEvaluationPipeline(
        client=client,
        max_chunks_per_dimension=args.max_chunks,
    )

    selected_chunks = pipeline.select_evidence(
        dimension=args.dimension,
        evidence_chunks=chunks,
    )

    print("\nSELECTED EVIDENCE")

    for index, chunk in enumerate(
        selected_chunks,
        start=1,
    ):
        token_count = chunker.count_tokens(
            chunk.text
        )

        print(
            f"[{index}] "
            f"id={chunk.chunk_id} "
            f"page={chunk.page} "
            f"section={chunk.section!r} "
            f"tokens={token_count}"
        )

    prompt = pipeline.build_prompt(
        paper_id=paper_id,
        dimension=args.dimension,
        evidence_chunks=selected_chunks,
    )

    print(
        f"\nPrompt characters: {len(prompt)}"
    )
    print(
        f"Evaluating dimension: {args.dimension}"
    )
    print(
        f"Model: {args.model or 'environment default'}"
    )

    started_at = time.perf_counter()

    raw_response = await client.generate(
        prompt
    )

    result = pipeline.parse_dimension_response(
        raw_response=raw_response,
        expected_dimension=args.dimension,
        allowed_chunk_ids={
            chunk.chunk_id
            for chunk in selected_chunks
        },
    )

    citation_client = OllamaCitationClient()
    citation_validator = CitationValidator(
        citation_client
    )

    evaluation_claims = build_evaluation_claims(
        paper_id=args.paper_id,
        evaluation=result,
        max_claims=2,
    )

    citation_validations = (
        await verify_evaluation_claims(
            validator=citation_validator,
            paper_id=args.paper_id,
            evaluation=result,
            evidence_chunks=selected_chunks,
            max_claims=2,
        )
    )

    claim_support_results = (
        build_claim_support_results(
            citation_validations
        )
    )

    result = pipeline._attach_verification(
        result=result,
        selected_chunks=selected_chunks,
        claim_support_results=claim_support_results,
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    print("\nCITATION VALIDATIONS")

    validations_by_claim_id = {
        validation.claim_id: validation
        for validation in citation_validations
    }

    for index, claim in enumerate(
        evaluation_claims,
        start=1,
    ):
        validation = validations_by_claim_id[
            claim.claim_id
        ]

        print(
            f"\n[{index}] Claim: {claim.text}"
        )

        print(
            json.dumps(
                validation.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )

    print("\nDIMENSION RESULT")

    print(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nElapsed seconds: "
        f"{elapsed_seconds:.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one real Ollama academic paper "
            "evaluation dimension."
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
        "--dimension",
        choices=REQUIRED_DIMENSIONS,
        default="methodological_rigor",
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

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(
        run_smoke_test(
            parse_args()
        )
    )