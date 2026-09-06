import asyncio
import json

import httpx
import pytest

from gpt_researcher.academic.ollama_claim_client import (
    CLAIM_RESPONSE_SCHEMA,
    OllamaClaimClient,
    OllamaClaimClientError,
)


def test_sends_structured_chat_request() -> None:
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
                    "content": '{"claims": []}',
                },
                "done": True,
            },
        )

    client = OllamaClaimClient(
        model="qwen3.5:2b",
        base_url="http://test-ollama:11434",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.generate("Extract claims from this evidence.")
    )

    assert result == '{"claims": []}'
    assert (
        captured_request["url"]
        == "http://test-ollama:11434/api/chat"
    )

    payload = captured_request["payload"]

    assert payload["model"] == "qwen3.5:2b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == CLAIM_RESPONSE_SCHEMA
    assert payload["options"]["temperature"] == 0.0
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"


def test_rejects_empty_prompt() -> None:
    client = OllamaClaimClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                request=request,
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="Prompt must not be empty",
    ):
        asyncio.run(client.generate("   "))


def test_handles_timeout() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Request timed out.",
            request=request,
        )

    client = OllamaClaimClient(
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        OllamaClaimClientError,
        match="timed out after 5.0 seconds",
    ):
        asyncio.run(client.generate("Extract claims."))


def test_handles_connection_error() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection refused.",
            request=request,
        )

    client = OllamaClaimClient(
        base_url="http://offline-ollama:11434",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        OllamaClaimClientError,
        match="Could not connect to Ollama",
    ):
        asyncio.run(client.generate("Extract claims."))


def test_handles_http_error() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            request=request,
            json={
                "error": "Internal server error",
            },
        )

    client = OllamaClaimClient(
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        OllamaClaimClientError,
        match="Ollama returned HTTP 500",
    ):
        asyncio.run(client.generate("Extract claims."))


def test_rejects_empty_model_content() -> None:
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

    client = OllamaClaimClient(
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        OllamaClaimClientError,
        match="empty claim content",
    ):
        asyncio.run(client.generate("Extract claims."))