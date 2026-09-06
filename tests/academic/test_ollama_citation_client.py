import asyncio
import json

import httpx
import pytest

from gpt_researcher.academic.ollama_citation_client import (
    CITATION_RESPONSE_SCHEMA,
    OllamaCitationClient,
    OllamaCitationClientError,
)


def test_sends_citation_validation_schema() -> None:
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
                            "status": "supported",
                            "explanation": (
                                "The evidence directly "
                                "supports the claim."
                            ),
                            "support_score": 0.95,
                        }
                    ),
                },
                "done": True,
            },
        )

    client = OllamaCitationClient(
        model="qwen3.5:9b",
        base_url="http://test-ollama:11434",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.generate(
            "Validate this claim against the evidence."
        )
    )

    parsed_result = json.loads(result)

    assert parsed_result["status"] == "supported"
    assert parsed_result["support_score"] == 0.95

    assert (
        captured_request["url"]
        == "http://test-ollama:11434/api/chat"
    )

    payload = captured_request["payload"]

    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["format"] == CITATION_RESPONSE_SCHEMA

    assert "status" in payload["format"]["properties"]
    assert "support_score" in payload["format"]["properties"]
    assert "claims" not in payload["format"]["properties"]

    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"

    system_prompt = payload["messages"][0]["content"].lower()

    assert "citation validation" in system_prompt
    assert "do not trust" in system_prompt


def test_uses_citation_model_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ACADEMIC_CITATION_MODEL",
        "qwen3.5:2b",
    )

    client = OllamaCitationClient()

    assert client.model == "qwen3.5:2b"


def test_rejects_empty_citation_content() -> None:
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

    client = OllamaCitationClient(
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        OllamaCitationClientError,
        match="empty citation validation content",
    ):
        asyncio.run(
            client.generate(
                "Validate this claim."
            )
        )