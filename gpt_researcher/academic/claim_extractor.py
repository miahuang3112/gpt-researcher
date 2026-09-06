from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import ValidationError

from gpt_researcher.academic.schemas import AcademicClaim, EvidenceChunk


class ClaimGenerationClient(Protocol):
    """
    主张生成客户端接口。

    后续可以接入 Ollama；单元测试中可以使用假的客户端，
    避免测试依赖真实模型。
    """

    async def generate(self, prompt: str) -> str:
        ...


class ClaimExtractor:
    """
    从证据片段中抽取结构化学术主张。
    """

    def __init__(
        self,
        client: ClaimGenerationClient,
        max_chunks: int = 20,
        max_chars_per_chunk: int = 4000,
    ) -> None:
        self.client = client
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk

    async def extract_claims(
        self,
        chunks: list[EvidenceChunk],
    ) -> list[AcademicClaim]:
        if not chunks:
            return []

        selected_chunks = chunks[: self.max_chunks]
        prompt = self._build_prompt(selected_chunks)
        raw_response = await self.client.generate(prompt)
        raw_claims = self._parse_response(raw_response)

        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in selected_chunks
        }

        claims: list[AcademicClaim] = []
        seen_claims: set[tuple[str, str]] = set()

        for item in raw_claims:
            claim = self._validate_claim(item, chunks_by_id)

            if claim is None:
                continue

            duplicate_key = (
                claim.paper_id,
                claim.text.strip().lower(),
            )

            if duplicate_key in seen_claims:
                continue

            seen_claims.add(duplicate_key)
            claims.append(claim)

        return claims

    def _build_prompt(
        self,
        chunks: list[EvidenceChunk],
    ) -> str:
        evidence = [
            {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "text": chunk.text[: self.max_chars_per_chunk],
            }
            for chunk in chunks
        ]

        evidence_json = json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
Extract factual academic claims from the evidence chunks below.

Allowed claim types:
- method
- result
- comparison
- limitation
- background
- other

Rules:
1. Every claim must be directly supported by the supplied evidence.
2. Every claim must reference at least one valid chunk_id.
3. Do not invent chunk IDs, results, numbers, methods, or conclusions.
4. A single claim may only reference chunks from the same paper.
5. Keep the claim language consistent with the evidence.
6. Confidence must be between 0 and 1.
7. Return JSON only, without Markdown.

Required JSON format:
{{
  "claims": [
    {{
      "text": "The factual academic claim",
      "claim_type": "result",
      "evidence_chunk_ids": ["chunk-1"],
      "confidence": 0.9
    }}
  ]
}}

Evidence chunks:
{evidence_json}
""".strip()

    def _parse_response(
        self,
        raw_response: str,
    ) -> list[dict[str, Any]]:
        text = raw_response.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []

        if isinstance(payload, dict):
            payload = payload.get("claims", [])

        if not isinstance(payload, list):
            return []

        return [
            item
            for item in payload
            if isinstance(item, dict)
        ]

    def _validate_claim(
        self,
        item: dict[str, Any],
        chunks_by_id: dict[str, EvidenceChunk],
    ) -> AcademicClaim | None:
        text = item.get("text")

        if not isinstance(text, str) or not text.strip():
            return None

        evidence_chunk_ids = item.get(
            "evidence_chunk_ids",
            [],
        )

        if not isinstance(evidence_chunk_ids, list):
            return None

        if not evidence_chunk_ids:
            return None

        if not all(
            isinstance(chunk_id, str)
            and chunk_id in chunks_by_id
            for chunk_id in evidence_chunk_ids
        ):
            return None

        unique_chunk_ids = list(
            dict.fromkeys(evidence_chunk_ids)
        )

        paper_ids = {
            chunks_by_id[chunk_id].paper_id
            for chunk_id in unique_chunk_ids
        }

        if len(paper_ids) != 1:
            return None

        paper_id = next(iter(paper_ids))
        supplied_paper_id = item.get("paper_id")

        if (
            supplied_paper_id is not None
            and supplied_paper_id != paper_id
        ):
            return None

        claim_id = self._create_claim_id(
            paper_id=paper_id,
            text=text.strip(),
            evidence_chunk_ids=unique_chunk_ids,
        )

        try:
            return AcademicClaim(
                claim_id=claim_id,
                paper_id=paper_id,
                text=text.strip(),
                claim_type=item.get("claim_type", "other"),
                evidence_chunk_ids=unique_chunk_ids,
                confidence=item.get("confidence", 0.0),
            )
        except ValidationError:
            return None

    @staticmethod
    def _create_claim_id(
        paper_id: str,
        text: str,
        evidence_chunk_ids: list[str],
    ) -> str:
        raw_value = "\0".join(
            [
                paper_id,
                text,
                *evidence_chunk_ids,
            ]
        )

        digest = hashlib.sha256(
            raw_value.encode("utf-8")
        ).hexdigest()[:16]

        return f"claim-{digest}"