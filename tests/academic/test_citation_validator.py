import asyncio
import json

import pytest

from gpt_researcher.academic.citation_validator import (
    CitationValidator,
    CitationValidatorError,
)
from gpt_researcher.academic.schemas import (
    AcademicClaim,
    EvidenceChunk,
)

from gpt_researcher.academic.ollama_structured_client import (
    OllamaStructuredClientError,
)


class FakeCitationClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt: str | None = None
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        self.calls += 1
        return self.response


def make_claim(
    evidence_chunk_ids: list[str] | None = None,
    paper_id: str = "paper-1",
) -> AcademicClaim:
    return AcademicClaim(
        claim_id="claim-1",
        paper_id=paper_id,
        text=(
            "The hybrid system achieved a Recall@10 "
            "of 0.84."
        ),
        claim_type="result",
        evidence_chunk_ids=(
            evidence_chunk_ids
            if evidence_chunk_ids is not None
            else ["chunk-1"]
        ),
        confidence=0.95,
    )


def make_chunk(
    chunk_id: str = "chunk-1",
    paper_id: str = "paper-1",
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Results",
        page=6,
        text=(
            "The hybrid retrieval system achieved "
            "a Recall@10 of 0.84."
        ),
    )


@pytest.mark.parametrize(
    ("status", "support_score"),
    [
        ("supported", 0.95),
        ("partially_supported", 0.65),
        ("insufficient_evidence", 0.30),
        ("contradicted", 0.05),
    ],
)
def test_accepts_valid_validation_result(
    status: str,
    support_score: float,
) -> None:
    response = json.dumps(
        {
            "status": status,
            "explanation": (
                "The validation result is based on "
                "the supplied evidence."
            ),
            "support_score": support_score,
        }
    )

    client = FakeCitationClient(response)
    validator = CitationValidator(client)

    result = asyncio.run(
        validator.validate_claim(
            claim=make_claim(),
            chunks=[make_chunk()],
        )
    )

    assert result.status == status
    assert result.support_score == support_score
    assert result.claim_id == "claim-1"
    assert result.paper_id == "paper-1"
    assert result.evidence_chunk_ids == ["chunk-1"]
    assert result.citation_id.startswith("citation-")

    assert client.calls == 1
    assert client.prompt is not None
    assert "Recall@10" in client.prompt
    assert "0.84" in client.prompt


def test_returns_not_found_for_missing_chunk() -> None:
    client = FakeCitationClient(
        '{"status": "supported"}'
    )

    validator = CitationValidator(client)

    result = asyncio.run(
        validator.validate_claim(
            claim=make_claim(
                evidence_chunk_ids=["missing-chunk"]
            ),
            chunks=[make_chunk()],
        )
    )

    assert result.status == "not_found"
    assert result.support_score == 0.0
    assert "missing-chunk" in result.explanation
    assert client.calls == 0


def test_rejects_chunk_from_different_paper() -> None:
    client = FakeCitationClient(
        '{"status": "supported"}'
    )

    validator = CitationValidator(client)

    result = asyncio.run(
        validator.validate_claim(
            claim=make_claim(paper_id="paper-1"),
            chunks=[
                make_chunk(
                    chunk_id="chunk-1",
                    paper_id="paper-2",
                )
            ],
        )
    )

    assert result.status == "not_found"
    assert result.support_score == 0.0
    assert "do not belong" in result.explanation
    assert client.calls == 0


def test_rejects_invalid_json() -> None:
    client = FakeCitationClient(
        "This is not valid JSON."
    )

    validator = CitationValidator(client)

    with pytest.raises(
        CitationValidatorError,
        match="returned invalid JSON",
    ):
        asyncio.run(
            validator.validate_claim(
                claim=make_claim(),
                chunks=[make_chunk()],
            )
        )


def test_rejects_missing_required_fields() -> None:
    response = json.dumps(
        {
            "status": "supported",
        }
    )

    client = FakeCitationClient(response)
    validator = CitationValidator(client)

    with pytest.raises(
        CitationValidatorError,
        match="missing required fields",
    ):
        asyncio.run(
            validator.validate_claim(
                claim=make_claim(),
                chunks=[make_chunk()],
            )
        )

class SequencedCitationClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        response = self.responses[
            self.call_count
        ]
        self.call_count += 1
        return response


def test_validate_claim_retries_invalid_json():
    client = SequencedCitationClient(
        [
            '{"status": "supported"',
            (
                '{"status": "supported", '
                '"explanation": "The evidence supports '
                'the claim.", '
                '"support_score": 0.9}'
            ),
        ]
    )
    validator = CitationValidator(client)

    result = asyncio.run(
        validator.validate_claim(
            make_claim(),
            [make_chunk()],
        )
    )

    assert client.call_count == 2
    assert result.status == "supported"
    assert result.support_score == 0.9


def test_validate_claim_stops_after_three_invalid_responses():
    client = SequencedCitationClient(
        [
            '{"status": "supported"',
            '{"status": "supported"',
            '{"status": "supported"',
        ]
    )
    validator = CitationValidator(client)

    with pytest.raises(
        CitationValidatorError,
        match="invalid JSON",
    ):
        asyncio.run(
            validator.validate_claim(
                make_claim(),
                [make_chunk()],
            )
        )

    assert client.call_count == 3

class TimeoutCitationClient:
    def __init__(
        self,
        *,
        timeout_count,
        success_response=None,
    ):
        self.timeout_count = timeout_count
        self.success_response = success_response
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1

        if self.call_count <= self.timeout_count:
            raise OllamaStructuredClientError(
                "Ollama request timed out."
            )

        return self.success_response


def test_validate_claim_retries_after_timeout():
    client = TimeoutCitationClient(
        timeout_count=1,
        success_response=(
            '{"status": "supported", '
            '"explanation": "The evidence supports '
            'the claim.", '
            '"support_score": 0.9}'
        ),
    )
    validator = CitationValidator(client)

    result = asyncio.run(
        validator.validate_claim(
            make_claim(),
            [make_chunk()],
        )
    )

    assert client.call_count == 2
    assert result.status == "supported"
    assert result.support_score == 0.9


def test_validate_claim_converts_three_timeouts():
    client = TimeoutCitationClient(
        timeout_count=3,
    )
    validator = CitationValidator(client)

    with pytest.raises(
        CitationValidatorError,
        match="failed after 3 attempts",
    ):
        asyncio.run(
            validator.validate_claim(
                make_claim(),
                [make_chunk()],
            )
        )

    assert client.call_count == 3