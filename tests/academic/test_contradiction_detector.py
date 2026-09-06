import asyncio
import json

import pytest

from gpt_researcher.academic.contradiction_detector import (
    ContradictionDetector,
    ContradictionDetectorError,
)
from gpt_researcher.academic.schemas import AcademicClaim


class FakeContradictionClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt: str | None = None
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        self.calls += 1
        return self.response


def make_claim(
    claim_id: str,
    paper_id: str,
    text: str,
) -> AcademicClaim:
    return AcademicClaim(
        claim_id=claim_id,
        paper_id=paper_id,
        text=text,
        claim_type="result",
        evidence_chunk_ids=[],
        confidence=0.9,
    )


@pytest.mark.parametrize(
    (
        "relation",
        "condition_differences",
        "confidence",
    ),
    [
        (
            "contradiction",
            [],
            0.95,
        ),
        (
            "partial_contradiction",
            [],
            0.80,
        ),
        (
            "different_conditions",
            ["dataset"],
            0.90,
        ),
        (
            "consistent",
            [],
            0.95,
        ),
        (
            "unrelated",
            [],
            0.90,
        ),
        (
            "uncertain",
            [],
            0.40,
        ),
    ],
)
def test_accepts_valid_relationship(
    relation: str,
    condition_differences: list[str],
    confidence: float,
) -> None:
    response = json.dumps(
        {
            "relation": relation,
            "condition_differences": (
                condition_differences
            ),
            "explanation": (
                "The relationship was determined from "
                "the two supplied claims."
            ),
            "confidence": confidence,
        }
    )

    client = FakeContradictionClient(response)
    detector = ContradictionDetector(client)

    claim_a = make_claim(
        claim_id="claim-a",
        paper_id="paper-a",
        text=(
            "Hybrid retrieval outperformed BM25."
        ),
    )

    claim_b = make_claim(
        claim_id="claim-b",
        paper_id="paper-b",
        text=(
            "BM25 outperformed hybrid retrieval."
        ),
    )

    result = asyncio.run(
        detector.compare_claims(
            claim_a=claim_a,
            claim_b=claim_b,
        )
    )

    assert result.relation == relation
    assert (
        result.condition_differences
        == condition_differences
    )
    assert result.confidence == confidence

    assert result.claim_a_id == "claim-a"
    assert result.claim_b_id == "claim-b"
    assert result.paper_a_id == "paper-a"
    assert result.paper_b_id == "paper-b"

    assert result.contradiction_id.startswith(
        "contradiction-"
    )

    assert client.calls == 1
    assert client.prompt is not None
    assert "Hybrid retrieval" in client.prompt
    assert "BM25 outperformed" in client.prompt


def test_identical_claims_are_consistent_without_model() -> None:
    client = FakeContradictionClient(
        "This response should not be used."
    )

    detector = ContradictionDetector(client)

    claim_a = make_claim(
        claim_id="claim-a",
        paper_id="paper-a",
        text="Hybrid retrieval improved Recall@10.",
    )

    claim_b = make_claim(
        claim_id="claim-b",
        paper_id="paper-b",
        text=(
            "  hybrid   retrieval improved "
            "Recall@10.  "
        ),
    )

    result = asyncio.run(
        detector.compare_claims(
            claim_a=claim_a,
            claim_b=claim_b,
        )
    )

    assert result.relation == "consistent"
    assert result.condition_differences == []
    assert result.confidence == 1.0
    assert client.calls == 0


def test_rejects_comparing_claim_with_itself() -> None:
    client = FakeContradictionClient("{}")
    detector = ContradictionDetector(client)

    claim = make_claim(
        claim_id="claim-a",
        paper_id="paper-a",
        text="Hybrid retrieval improved Recall@10.",
    )

    with pytest.raises(
        ValueError,
        match="cannot be compared with itself",
    ):
        asyncio.run(
            detector.compare_claims(
                claim_a=claim,
                claim_b=claim,
            )
        )

    assert client.calls == 0


def test_rejects_claims_from_same_paper() -> None:
    client = FakeContradictionClient("{}")
    detector = ContradictionDetector(client)

    claim_a = make_claim(
        claim_id="claim-a",
        paper_id="paper-a",
        text="Claim A.",
    )

    claim_b = make_claim(
        claim_id="claim-b",
        paper_id="paper-a",
        text="Claim B.",
    )

    with pytest.raises(
        ValueError,
        match="requires claims from different papers",
    ):
        asyncio.run(
            detector.compare_claims(
                claim_a=claim_a,
                claim_b=claim_b,
            )
        )

    assert client.calls == 0


def test_rejects_invalid_json() -> None:
    client = FakeContradictionClient(
        "This is not valid JSON."
    )

    detector = ContradictionDetector(client)

    with pytest.raises(
        ContradictionDetectorError,
        match="returned invalid JSON",
    ):
        asyncio.run(
            detector.compare_claims(
                claim_a=make_claim(
                    claim_id="claim-a",
                    paper_id="paper-a",
                    text="Claim A.",
                ),
                claim_b=make_claim(
                    claim_id="claim-b",
                    paper_id="paper-b",
                    text="Claim B.",
                ),
            )
        )


def test_rejects_missing_required_fields() -> None:
    response = json.dumps(
        {
            "relation": "contradiction",
        }
    )

    client = FakeContradictionClient(response)
    detector = ContradictionDetector(client)

    with pytest.raises(
        ContradictionDetectorError,
        match="missing required fields",
    ):
        asyncio.run(
            detector.compare_claims(
                claim_a=make_claim(
                    claim_id="claim-a",
                    paper_id="paper-a",
                    text="Claim A.",
                ),
                claim_b=make_claim(
                    claim_id="claim-b",
                    paper_id="paper-b",
                    text="Claim B.",
                ),
            )
        )