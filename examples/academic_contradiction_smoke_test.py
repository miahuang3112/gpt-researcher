import asyncio
import json
import time
from typing import Any

from gpt_researcher.academic.contradiction_detector import (
    ContradictionDetector,
)
from gpt_researcher.academic.ollama_contradiction_client import (
    OllamaContradictionClient,
)
from gpt_researcher.academic.schemas import AcademicClaim


def model_to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def make_claim(
    claim_id: str,
    paper_id: str,
    text: str,
) -> AcademicClaim:
    return AcademicClaim(
        claim_id=claim_id,
        paper_id=paper_id,
        text=text,
        claim_type="comparison",
        evidence_chunk_ids=[],
        confidence=0.9,
    )


async def run_case(
    detector: ContradictionDetector,
    label: str,
    claim_a: AcademicClaim,
    claim_b: AcademicClaim,
    expected_relation: str,
) -> Any:
    print(f"\nCase: {label}")
    print(f"Claim A: {claim_a.text}")
    print(f"Claim B: {claim_b.text}")

    started_at = time.perf_counter()

    result = await detector.compare_claims(
        claim_a=claim_a,
        claim_b=claim_b,
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

    if result.relation != expected_relation:
        raise RuntimeError(
            f"Expected relation "
            f"'{expected_relation}', "
            f"but received '{result.relation}'."
        )

    if (
        expected_relation == "different_conditions"
        and not result.condition_differences
    ):
        raise RuntimeError(
            "The detector identified different conditions "
            "but did not record any condition differences."
        )

    return result


async def main() -> None:
    client = OllamaContradictionClient()
    detector = ContradictionDetector(client)

    print(f"Contradiction model: {client.model}")
    print(f"Ollama URL: {client.base_url}")

    contradiction_a = make_claim(
        claim_id="claim-direct-a",
        paper_id="paper-direct-a",
        text=(
            "Using the same released Hybrid-v1 and BM25-v1 "
            "implementations without retraining or parameter "
            "changes, the identical Benchmark-X test split, "
            "queries, relevance judgements, and official "
            "Recall@10 protocol, Hybrid-v1 achieved 0.84 "
            "while BM25-v1 achieved 0.71. Hybrid-v1 "
            "outperformed BM25-v1."
        ),
    )

    contradiction_b = make_claim(
        claim_id="claim-direct-b",
        paper_id="paper-direct-b",
        text=(
            "Using the same released Hybrid-v1 and BM25-v1 "
            "implementations without retraining or parameter "
            "changes, the identical Benchmark-X test split, "
            "queries, relevance judgements, and official "
            "Recall@10 protocol, Hybrid-v1 achieved 0.68 "
            "while BM25-v1 achieved 0.79. BM25-v1 "
            "outperformed Hybrid-v1."
        ),
    )

    await run_case(
        detector=detector,
        label="direct contradiction",
        claim_a=contradiction_a,
        claim_b=contradiction_b,
        expected_relation="contradiction",
    )

    different_conditions_a = make_claim(
        claim_id="claim-condition-a",
        paper_id="paper-condition-a",
        text=(
            "On an English biomedical article dataset, "
            "hybrid retrieval achieved Recall@10 of 0.84 "
            "and outperformed BM25 at 0.71."
        ),
    )

    different_conditions_b = make_claim(
        claim_id="claim-condition-b",
        paper_id="paper-condition-b",
        text=(
            "On a Chinese news article dataset, BM25 "
            "achieved Recall@10 of 0.82 and outperformed "
            "hybrid retrieval at 0.75."
        ),
    )

    await run_case(
        detector=detector,
        label="different experimental conditions",
        claim_a=different_conditions_a,
        claim_b=different_conditions_b,
        expected_relation="different_conditions",
    )

    consistent_a = make_claim(
        claim_id="claim-consistent-a",
        paper_id="paper-consistent-a",
        text=(
            "On Benchmark-X, hybrid retrieval achieved "
            "a higher Recall@10 than BM25."
        ),
    )

    consistent_b = make_claim(
        claim_id="claim-consistent-b",
        paper_id="paper-consistent-b",
        text=(
            "Using the official Benchmark-X evaluation "
            "protocol, hybrid retrieval outperformed "
            "BM25 in Recall@10."
        ),
    )

    await run_case(
        detector=detector,
        label="consistent conclusions",
        claim_a=consistent_a,
        claim_b=consistent_b,
        expected_relation="consistent",
    )

    print("\nAll contradiction smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())