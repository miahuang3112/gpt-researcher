from __future__ import annotations

import os
from typing import Any

import httpx


CLAIM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                    },
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "method",
                            "result",
                            "comparison",
                            "limitation",
                            "background",
                            "other",
                        ],
                    },
                    "evidence_chunk_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "minItems": 1,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [
                    "text",
                    "claim_type",
                    "evidence_chunk_ids",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class OllamaClaimClientError(RuntimeError):
    """
    Ollama Claim抽取请求失败。
    """


class OllamaClaimClient:
    """
    通过本地Ollama Chat API生成结构化Claim JSON。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        think: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = (
            model
            or os.getenv(
                "ACADEMIC_CLAIM_MODEL",
                "qwen3.5:2b",
            )
        )

        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")

        self.timeout = timeout
        self.temperature = temperature
        self.think = think
        self.transport = transport

    async def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an academic evidence extraction "
                        "component. Extract only claims directly "
                        "supported by the supplied evidence. "
                        "Return JSON matching the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "think": self.think,
            "format": CLAIM_RESPONSE_SCHEMA,
            "options": {
                "temperature": self.temperature,
            },
            "keep_alive": "5m",
        }

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    "/api/chat",
                    json=payload,
                )

                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise OllamaClaimClientError(
                f"Ollama request timed out after "
                f"{self.timeout} seconds."
            ) from exc

        except httpx.ConnectError as exc:
            raise OllamaClaimClientError(
                f"Could not connect to Ollama at "
                f"{self.base_url}."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise OllamaClaimClientError(
                f"Ollama returned HTTP "
                f"{exc.response.status_code}."
            ) from exc

        except httpx.HTTPError as exc:
            raise OllamaClaimClientError(
                f"Ollama request failed: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaClaimClientError(
                "Ollama returned invalid response JSON."
            ) from exc

        message = data.get("message")

        if not isinstance(message, dict):
            raise OllamaClaimClientError(
                "Ollama response is missing message."
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise OllamaClaimClientError(
                "Ollama returned empty claim content."
            )

        return content.strip()