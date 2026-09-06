from __future__ import annotations

import os
from typing import Any

import httpx

from gpt_researcher.academic.ollama_structured_client import (
    OllamaStructuredClient,
    OllamaStructuredClientError,
)


CITATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "supported",
                "partially_supported",
                "contradicted",
                "insufficient_evidence",
            ],
        },
        "explanation": {
            "type": "string",
        },
        "support_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": [
        "status",
        "explanation",
        "support_score",
    ],
    "additionalProperties": False,
}


OllamaCitationClientError = OllamaStructuredClientError


class OllamaCitationClient(OllamaStructuredClient):
    """
    通过本地Ollama验证Claim是否被Evidence支持。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
        temperature: float = 0.0,
        think: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            model=(
                model
                or os.getenv(
                    "ACADEMIC_CITATION_MODEL",
                    "qwen3.5:9b",
                )
            ),
            response_schema=CITATION_RESPONSE_SCHEMA,
            system_prompt=(
                "You are an academic citation validation "
                "component. Determine whether each claim "
                "is supported by the supplied evidence. "
                "Check numbers, comparisons, negation, "
                "scope and experimental conditions. "
                "Do not trust the claim's confidence score. "
                "Return JSON matching the requested schema."
            ),
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            think=think,
            response_name="citation validation",
            transport=transport,
        )