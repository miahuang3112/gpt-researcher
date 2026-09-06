import asyncio
import json
import time
from typing import Any

from gpt_researcher.academic.claim_extractor import (
    ClaimExtractor,
)
from gpt_researcher.academic.ollama_claim_client import (
    OllamaClaimClient,
)
from gpt_researcher.academic.schemas import EvidenceChunk


def model_to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


async def main() -> None:
    chunks = [
        EvidenceChunk(
            chunk_id="demo-chunk-1",
            paper_id="demo-paper-1",
            section="Methods",
            page=3,
            text=(
                "The study compared BM25, dense retrieval, "
                "and a hybrid retrieval system on 1,000 "
                "manually labelled research queries. "
                "Every system returned the top 10 passages."
            ),
        ),
        EvidenceChunk(
            chunk_id="demo-chunk-2",
            paper_id="demo-paper-1",
            section="Results",
            page=6,
            text=(
                "The hybrid retrieval system achieved a "
                "Recall@10 of 0.84. BM25 achieved 0.71, "
                "while dense retrieval achieved 0.77."
            ),
        ),
        EvidenceChunk(
            chunk_id="demo-chunk-3",
            paper_id="demo-paper-1",
            section="Limitations",
            page=8,
            text=(
                "The experiments included only English "
                "biomedical articles. The study did not "
                "evaluate retrieval latency."
            ),
        ),
    ]

    client = OllamaClaimClient()
    extractor = ClaimExtractor(client)

    print(f"Model: {client.model}")
    print(f"Ollama URL: {client.base_url}")
    print("Running real Claim extraction...")

    started_at = time.perf_counter()

    claims = await extractor.extract_claims(chunks)

    elapsed = time.perf_counter() - started_at

    if not claims:
        raise RuntimeError(
            "Ollama returned no valid academic claims."
        )

    valid_chunk_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    for claim in claims:
        assert set(
            claim.evidence_chunk_ids
        ).issubset(valid_chunk_ids)

    print(f"\nExtracted claims: {len(claims)}")
    print(f"Elapsed seconds: {elapsed:.2f}")

    for index, claim in enumerate(claims, start=1):
        print(f"\nClaim {index}:")
        print(
            json.dumps(
                model_to_dict(claim),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())