import json

import httpx
import pytest

from gpt_researcher.academic.ollama_paper_evaluator_client import (
    OllamaPaperEvaluatorClient,
    OllamaPaperEvaluatorClientError,
    PAPER_DIMENSION_RESPONSE_SCHEMA,
)


VALID_DIMENSION_RESPONSE = {
    "dimension": "methodological_rigor",
    "status": "scored",
    "score": 4.0,
    "rationale": (
        "The method is clearly defined and supported "
        "by implementation details."
    ),
    "evidence_chunk_ids": [
        "chunk-methods-001"
    ],
    "strengths": [
        "The algorithm is described step by step."
    ],
    "concerns": [
        "The sensitivity analysis is limited."
    ],
}


def test_schema_bounds_free_text_to_prevent_truncated_json() -> None:
    properties = PAPER_DIMENSION_RESPONSE_SCHEMA["properties"]

    assert "confidence" not in properties
    assert properties["rationale"]["maxLength"] == 600
    assert properties["strengths"]["maxItems"] == 2
    assert properties["concerns"]["maxItems"] == 2


@pytest.mark.asyncio
async def test_client_uses_paper_dimension_schema():
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(
            request.content.decode("utf-8")
        )

        assert request.url.path == "/api/chat"
        assert payload["model"] == "test-evaluator-model"
        assert payload["stream"] is False
        assert payload["options"]["num_predict"] == 1024

        assert (
            payload["format"]
            == PAPER_DIMENSION_RESPONSE_SCHEMA
        )

        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "content": json.dumps(
                        VALID_DIMENSION_RESPONSE
                    )
                }
            },
        )

    client = OllamaPaperEvaluatorClient(
        model="test-evaluator-model",
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    raw_response = await client.generate(
        "Evaluate methodological rigor."
    )

    parsed_response = json.loads(raw_response)

    assert (
        parsed_response["dimension"]
        == "methodological_rigor"
    )
    assert parsed_response["status"] == "scored"
    assert parsed_response["score"] == 4.0

    assert parsed_response["evidence_chunk_ids"] == [
        "chunk-methods-001"
    ]


@pytest.mark.asyncio
async def test_client_uses_environment_model(
    monkeypatch,
):
    monkeypatch.setenv(
        "ACADEMIC_PAPER_EVALUATOR_MODEL",
        "environment-evaluator-model",
    )

    observed_model: str | None = None

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal observed_model

        payload = json.loads(
            request.content.decode("utf-8")
        )

        observed_model = payload["model"]

        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "content": json.dumps(
                        VALID_DIMENSION_RESPONSE
                    )
                }
            },
        )

    client = OllamaPaperEvaluatorClient(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    await client.generate(
        "Evaluate methodological rigor."
    )

    assert (
        observed_model
        == "environment-evaluator-model"
    )


@pytest.mark.asyncio
async def test_client_rejects_empty_response():
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "message": {
                    "content": ""
                }
            },
        )

    client = OllamaPaperEvaluatorClient(
        model="test-evaluator-model",
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        OllamaPaperEvaluatorClientError,
        match="empty",
    ):
        await client.generate(
            "Evaluate methodological rigor."
        )
