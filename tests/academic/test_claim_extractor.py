import asyncio
import json

from gpt_researcher.academic.claim_extractor import ClaimExtractor
from gpt_researcher.academic.schemas import EvidenceChunk


class FakeClaimClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def make_chunk(
    chunk_id: str = "chunk-1",
    paper_id: str = "paper-1",
    text: str = "The proposed method improved accuracy by 10 percent.",
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        section="Results",
        page=3,
    )


def test_extracts_valid_claim() -> None:
    response = json.dumps(
        {
            "claims": [
                {
                    "text": "The proposed method improved accuracy by 10 percent.",
                    "claim_type": "result",
                    "evidence_chunk_ids": ["chunk-1"],
                    "confidence": 0.9,
                }
            ]
        }
    )

    client = FakeClaimClient(response)
    extractor = ClaimExtractor(client)

    claims = asyncio.run(
        extractor.extract_claims([make_chunk()])
    )

    assert len(claims) == 1
    assert claims[0].paper_id == "paper-1"
    assert claims[0].claim_type == "result"
    assert claims[0].evidence_chunk_ids == ["chunk-1"]
    assert claims[0].confidence == 0.9
    assert claims[0].claim_id.startswith("claim-")

    assert client.prompt is not None
    assert "chunk-1" in client.prompt


def test_rejects_unknown_chunk_id() -> None:
    response = json.dumps(
        {
            "claims": [
                {
                    "text": "An unsupported result.",
                    "claim_type": "result",
                    "evidence_chunk_ids": ["invented-chunk"],
                    "confidence": 0.8,
                }
            ]
        }
    )

    extractor = ClaimExtractor(
        FakeClaimClient(response)
    )

    claims = asyncio.run(
        extractor.extract_claims([make_chunk()])
    )

    assert claims == []


def test_rejects_claim_using_multiple_papers() -> None:
    response = json.dumps(
        {
            "claims": [
                {
                    "text": "The two papers produced the same result.",
                    "claim_type": "comparison",
                    "evidence_chunk_ids": [
                        "chunk-1",
                        "chunk-2",
                    ],
                    "confidence": 0.7,
                }
            ]
        }
    )

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            paper_id="paper-1",
        ),
        make_chunk(
            chunk_id="chunk-2",
            paper_id="paper-2",
        ),
    ]

    extractor = ClaimExtractor(
        FakeClaimClient(response)
    )

    claims = asyncio.run(
        extractor.extract_claims(chunks)
    )

    assert claims == []


def test_removes_duplicate_claims() -> None:
    claim = {
        "text": "The proposed method improved accuracy.",
        "claim_type": "result",
        "evidence_chunk_ids": ["chunk-1"],
        "confidence": 0.8,
    }

    response = json.dumps(
        {
            "claims": [
                claim,
                claim,
            ]
        }
    )

    extractor = ClaimExtractor(
        FakeClaimClient(response)
    )

    claims = asyncio.run(
        extractor.extract_claims([make_chunk()])
    )

    assert len(claims) == 1


def test_returns_empty_list_for_invalid_json() -> None:
    extractor = ClaimExtractor(
        FakeClaimClient("This is not JSON")
    )

    claims = asyncio.run(
        extractor.extract_claims([make_chunk()])
    )

    assert claims == []