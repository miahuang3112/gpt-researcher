from __future__ import annotations

import os
from typing import Any

import httpx

from gpt_researcher.academic.ollama_structured_client import (
    OllamaStructuredClient,
    OllamaStructuredClientError,
)


CONTRADICTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": [
                "contradiction",
                "partial_contradiction",
                "different_conditions",
                "consistent",
                "unrelated",
                "uncertain",
            ],
        },
        "condition_differences": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "population",
                    "dataset",
                    "metric",
                    "method",
                    "intervention",
                    "baseline",
                    "time_period",
                    "sample_size",
                    "evaluation_setting",
                    "other",
                ],
            },
        },
        "explanation": {
            "type": "string",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": [
        "relation",
        "condition_differences",
        "explanation",
        "confidence",
    ],
    "additionalProperties": False,
}


OllamaContradictionClientError = (
    OllamaStructuredClientError
)


class OllamaContradictionClient(
    OllamaStructuredClient
):
    """
    通过本地Ollama判断跨论文Claim之间的关系。
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
                    "ACADEMIC_CONTRADICTION_MODEL",
                    "qwen3.5:9b",
                )
            ),
            response_schema=(
                CONTRADICTION_RESPONSE_SCHEMA
            ),
            system_prompt=(
                "You are an academic contradiction "
                "analysis component. Compare claims from "
                "different papers and determine whether "
                "they truly contradict each other, are "
                "consistent, are unrelated, or only differ "
                "because of experimental conditions. "
                "Check datasets, populations, metrics, "
                "methods, baselines, time periods and "
                "evaluation settings." 
                "Do not treat different numbers alone as a "
                "contradiction. Only report a condition "
                "difference when it is explicitly stated in "
                "the supplied claims. Different paper IDs are "
                "not evidence of different implementations or "
                "training procedures. Never invent unstated "
                "experimental differences. Return JSON matching "
                "the requested schema."
            ),
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            think=think,
            response_name="contradiction detection",
            transport=transport,
        )