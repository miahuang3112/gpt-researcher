from __future__ import annotations

import os
from typing import Any

import httpx

from gpt_researcher.academic.ollama_structured_client import (
    OllamaStructuredClient,
    OllamaStructuredClientError,
)

PAPER_DIMENSION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dimension": {
            "type": "string",
            "enum": [
                "research_question_clarity",
                "methodological_rigor",
                "experimental_design",
                "baseline_quality",
                "result_validity",
                "reproducibility",
                "novelty",
                "limitations_discussion",
                "citation_quality",
            ],
        },
        "status": {
            "type": "string",
            "enum": [
                "scored",
                "insufficient_evidence",
            ],
        },
        "score": {
            "anyOf": [
                {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 5.0,
                },
                {
                    "type": "null",
                },
            ],
        },
        "rationale": {
            "type": "string",
            "maxLength": 600,
        },
        "evidence_chunk_ids": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "uniqueItems": True,
        },
        "external_evidence_ids": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "uniqueItems": True,
        },
        "strengths": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "string",
                "maxLength": 240,
            },
        },
        "concerns": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "string",
                "maxLength": 240,
            },
        },
    },
    "required": [
        "dimension",
        "status",
        "score",
        "rationale",
        "evidence_chunk_ids",
        "strengths",
        "concerns",
    ],
    "additionalProperties": False,
}

OllamaPaperEvaluatorClientError = OllamaStructuredClientError


class OllamaPaperEvaluatorClient(OllamaStructuredClient):
    """
    通过本地Ollama为论文的单个评价维度
    生成结构化候选评分。

    该客户端只负责模型调用和JSON约束。
    Evidence ID验证、权重和总分由PaperEvaluator处理。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
        temperature: float = 0.0,
        think: bool = False,
        num_predict: int = 1024,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            model=(
                model
                or os.getenv(
                    "ACADEMIC_PAPER_EVALUATOR_MODEL",
                    "qwen3.5:9b",
                )
            ),
            response_schema=PAPER_DIMENSION_RESPONSE_SCHEMA,
            system_prompt=(
                "You are an academic paper evaluator "
                "component. Evaluate each dimension "
                "of the paper based on the supplied evidence. "
                "Check numbers, comparisons, negation, "
                "scope and experimental conditions. "
                "Do not trust the claim's confidence score. "
                "Return JSON matching the requested schema."
            ),
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            think=think,
            num_predict=num_predict,
            response_name="paper evaluation",
            transport=transport,
        )
