import asyncio
import json
import time
from typing import Any

from gpt_researcher.academic.citation_validator import (
    CitationValidator,
)
from gpt_researcher.academic.ollama_citation_client import (
    OllamaCitationClient,
)
from gpt_researcher.academic.schemas import (
    AcademicClaim,
    EvidenceChunk,
)


def model_to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


async def validate_and_print(
    validator: CitationValidator,
    claim: AcademicClaim,
    chunks: list[EvidenceChunk],
    label: str,
) -> Any:
    print(f"\nValidating: {label}")
    print(f"Claim: {claim.text}")
    print(f"Claim confidence: {claim.confidence}")

    started_at = time.perf_counter()

    result = await validator.validate_claim(
        claim=claim,
        chunks=chunks,
    )

    elapsed = time.perf_counter() - started_at

    print(f"Elapsed seconds: {elapsed:.2f}")
    print(
        json.dumps(
            model_to_dict(result),
            ensure_ascii=False,
            indent=2,
        )
    )

    return result


async def main() -> None:
    evidence = EvidenceChunk(
        chunk_id="demo-chunk-2",
        paper_id="demo-paper-1",
        section="Results",
        page=6,
        text=(
            "The hybrid retrieval system achieved a "
            "Recall@10 of 0.84. BM25 achieved 0.71, "
            "while dense retrieval achieved 0.77."
        ),
    )

    incorrect_claim = AcademicClaim(
        claim_id="claim-incorrect-comparison",
        paper_id="demo-paper-1",
        text=(
            "The hybrid retrieval system achieved a "
            "Recall@10 of 0.84, which was higher than "
            "BM25 at 0.71 but lower than dense retrieval "
            "at 0.77."
        ),
        claim_type="comparison",
        evidence_chunk_ids=["demo-chunk-2"],
        confidence=1.0,
    )

    correct_claim = AcademicClaim(
        claim_id="claim-correct-comparison",
        paper_id="demo-paper-1",
        text=(
            "The hybrid retrieval system achieved a "
            "higher Recall@10 than both BM25 and dense "
            "retrieval."
        ),
        claim_type="comparison",
        evidence_chunk_ids=["demo-chunk-2"],
        confidence=0.95,
    )

    client = OllamaCitationClient()
    validator = CitationValidator(client)

    print(f"Citation model: {client.model}")
    print(f"Ollama URL: {client.base_url}")

    incorrect_result = await validate_and_print(
        validator=validator,
        claim=incorrect_claim,
        chunks=[evidence],
        label="incorrect numerical comparison",
    )

    correct_result = await validate_and_print(
        validator=validator,
        claim=correct_claim,
        chunks=[evidence],
        label="correct numerical comparison",
    )

    if incorrect_result.status != "contradicted":
        raise RuntimeError(
            "The validator failed to identify the "
            "incorrect numerical comparison."
        )

    if incorrect_result.support_score > 0.15:
        raise RuntimeError(
            "The contradicted claim received an "
            "unexpectedly high support score."
        )

    if correct_result.status != "supported":
        raise RuntimeError(
            "The validator failed to accept the "
            "correct numerical comparison."
        )

    if correct_result.support_score < 0.85:
        raise RuntimeError(
            "The supported claim received an "
            "unexpectedly low support score."
        )

    print("\nCitation smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
