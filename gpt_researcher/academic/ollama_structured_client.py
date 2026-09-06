from __future__ import annotations

import os
from typing import Any

import httpx

class OllamaStructuredClientError(RuntimeError):
    """
    Ollama结构化请求失败。
    """

class OllamaStructuredClient:
    """
    调用Ollama并要求模型按照JSON Schema输出。
    """

    def __init__(
        self,
        model: str,
        response_schema: dict[str, Any],
        system_prompt: str,
        base_url: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        think: bool = False,
        num_predict: int | None = None,
        response_name: str = "structured",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.response_schema = response_schema
        self.system_prompt = system_prompt

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
        self.num_predict = num_predict
        self.response_name = response_name
        self.transport = transport

    async def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "think": self.think,
            "format": self.response_schema,
            "options": {
                "temperature": self.temperature,
            },
            "keep_alive": "5m",
        }
        if self.num_predict is not None:
            payload["options"]["num_predict"] = self.num_predict

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
            raise OllamaStructuredClientError(
                f"Ollama request timed out after "
                f"{self.timeout} seconds."
            ) from exc

        except httpx.ConnectError as exc:
            raise OllamaStructuredClientError(
                f"Could not connect to Ollama at "
                f"{self.base_url}."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise OllamaStructuredClientError(
                f"Ollama returned HTTP "
                f"{exc.response.status_code}."
            ) from exc

        except httpx.HTTPError as exc:
            raise OllamaStructuredClientError(
                f"Ollama request failed: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaStructuredClientError(
                "Ollama returned invalid response JSON."
            ) from exc

        message = data.get("message")

        if not isinstance(message, dict):
            raise OllamaStructuredClientError(
                "Ollama response is missing message."
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise OllamaStructuredClientError(
                "Ollama returned empty "
                f"{self.response_name} content."
            )

        return content.strip()
