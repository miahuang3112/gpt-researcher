import asyncio
import json

import httpx
import pytest

from gpt_researcher.academic.ollama_contradiction_client import (
    CONTRADICTION_RESPONSE_SCHEMA,
    OllamaContradictionClient,
    OllamaContradictionClientError,
)


def test_sends_contradiction_schema() -> None:
    captured_request: dict = {}

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["payload"] = json.loads(
            request.content.decode("utf-8")
        )

        return httpx.Response(
            status_code=200,
            request=request,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "relation": (
                                "different_conditions"
                            ),
                            "condition_differences": [
                                "dataset"
                            ],
                            "explanation": (
                                "The claims use different "
                                "datasets."
                            ),
                            "confidence": 0.92,
                        }
                    ),
                },
                "done": True,
            },
        )

    client = OllamaContradictionClient(
        model="qwen3.5:9b",
        base_url="http://test-ollama:11434",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.generate(
            "Compare these two academic claims."
        )
    )

    parsed_result = json.loads(result)

    assert (
        parsed_result["relation"]
        == "different_conditions"
    )
    assert parsed_result[
        "condition_differences"
    ] == ["dataset"]
    assert parsed_result["confidence"] == 0.92

    assert (
        captured_request["url"]
        == "http://test-ollama:11434/api/chat"
    )

    payload = captured_request["payload"]

    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0.0
    assert (
        payload["format"]
        == CONTRADICTION_RESPONSE_SCHEMA
    )

    properties = payload["format"]["properties"]

    assert "relation" in properties
    assert "condition_differences" in properties
    assert "confidence" in properties

    assert "claims" not in properties
    assert "status" not in properties
    assert "support_score" not in properties

    relation_values = properties["relation"]["enum"]

    assert "contradiction" in relation_values
    assert "different_conditions" in relation_values
    assert "consistent" in relation_values
    assert "unrelated" in relation_values

    system_prompt = (
        payload["messages"][0]["content"].lower()
    )

    assert "contradiction" in system_prompt
    assert "experimental conditions" in system_prompt
    assert "different numbers" in system_prompt


def test_uses_contradiction_model_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ACADEMIC_CONTRADICTION_MODEL",
        "qwen3.5:2b",
    )

    client = OllamaContradictionClient()

    assert client.model == "qwen3.5:2b"


def test_rejects_empty_contradiction_content() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            request=request,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                }
            },
        )

    client = OllamaContradictionClient(
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        OllamaContradictionClientError,
        match="empty contradiction detection content",
    ):
        asyncio.run(
            client.generate(
                "Compare these claims."
            )
        )