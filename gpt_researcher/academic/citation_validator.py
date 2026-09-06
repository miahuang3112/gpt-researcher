from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import ValidationError

from gpt_researcher.academic.schemas import (
    AcademicClaim,
    CitationValidation,
    EvidenceChunk,
)

from gpt_researcher.academic.ollama_structured_client import (
    OllamaStructuredClientError,
)

class CitationValidationClient(Protocol):
    """
    引用验证模型客户端接口。
    """

    async def generate(self, prompt: str) -> str:
        ...


class CitationValidatorError(RuntimeError):
    """
    Citation Validator无法完成验证。
    """


class CitationValidator:
    """
    检查AcademicClaim是否被EvidenceChunk支持。
    """

    def __init__(
        self,
        client: CitationValidationClient,
        max_chars_per_chunk: int = 4000,
    ) -> None:
        self.client = client
        self.max_chars_per_chunk = max_chars_per_chunk

    async def validate_claim(
        self,
        claim: AcademicClaim,
        chunks: list[EvidenceChunk],
    ) -> CitationValidation:
        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

        requested_chunk_ids = list(
            dict.fromkeys(claim.evidence_chunk_ids)
        )

        if not requested_chunk_ids:
            return self._not_found_result(
                claim=claim,
                evidence_chunk_ids=[],
                explanation=(
                    "The claim does not reference any "
                    "evidence chunks."
                ),
            )

        missing_chunk_ids = [
            chunk_id
            for chunk_id in requested_chunk_ids
            if chunk_id not in chunks_by_id
        ]

        if missing_chunk_ids:
            missing_text = ", ".join(missing_chunk_ids)

            return self._not_found_result(
                claim=claim,
                evidence_chunk_ids=requested_chunk_ids,
                explanation=(
                    "The following evidence chunks were "
                    f"not found: {missing_text}."
                ),
            )

        selected_chunks = [
            chunks_by_id[chunk_id]
            for chunk_id in requested_chunk_ids
        ]

        mismatched_chunks = [
            chunk.chunk_id
            for chunk in selected_chunks
            if chunk.paper_id != claim.paper_id
        ]

        if mismatched_chunks:
            mismatch_text = ", ".join(mismatched_chunks)

            return self._not_found_result(
                claim=claim,
                evidence_chunk_ids=requested_chunk_ids,
                explanation=(
                    "The following evidence chunks do not "
                    "belong to the claim's paper: "
                    f"{mismatch_text}."
                ),
            )

        char_limits = (
            self.max_chars_per_chunk,
            max(1, self.max_chars_per_chunk // 2),
            max(1, self.max_chars_per_chunk // 4),
        )
        max_attempts = len(char_limits)

        for attempt, char_limit in enumerate(
            char_limits,
            start=1,
        ):
            prompt = self._build_prompt(
                claim=claim,
                chunks=selected_chunks,
                max_chars_per_chunk=char_limit,
            )

            try:
                raw_response = await self.client.generate(
                    prompt
                )
                result = self._parse_response(
                    raw_response
                )
                break
            except CitationValidatorError:
                if attempt == max_attempts:
                    raise

                print(
                    "Citation validation returned "
                    "an invalid response; retrying "
                    f"({attempt + 1}/{max_attempts}), "
                    f"reducing evidence to "
                    f"{char_limits[attempt]} characters per chunk..."
                )
            except OllamaStructuredClientError as exc:
                if attempt == max_attempts:
                    raise CitationValidatorError(
                        "Citation validation request "
                        "failed after 3 attempts."
                    ) from exc

                print(
                    "Citation validation request failed; retrying "
                    f"({attempt + 1}/{max_attempts}), "
                    "reducing evidence to "
                    f"{char_limits[attempt]} characters per chunk: "
                    f"{exc}"
                )

        required_fields = {
            "status",
            "explanation",
            "support_score",
        }

        missing_fields = required_fields.difference(result)

        if missing_fields:
            missing_text = ", ".join(
                sorted(missing_fields)
            )

            raise CitationValidatorError(
                "Citation validation response is missing "
                f"required fields: {missing_text}."
            )

        try:
            return CitationValidation(
                citation_id=self._create_citation_id(
                    claim_id=claim.claim_id,
                    evidence_chunk_ids=requested_chunk_ids,
                ),
                claim_id=claim.claim_id,
                paper_id=claim.paper_id,
                evidence_chunk_ids=requested_chunk_ids,
                status=result["status"],
                explanation=result["explanation"],
                support_score=result["support_score"],
            )
        except ValidationError as exc:
            raise CitationValidatorError(
                "Citation validation response failed "
                "schema validation."
            ) from exc

    def _build_prompt(
        self,
        claim: AcademicClaim,
        chunks: list[EvidenceChunk],
        *,
        max_chars_per_chunk: int | None = None,
    ) -> str:
        char_limit = (
            self.max_chars_per_chunk
            if max_chars_per_chunk is None
            else max_chars_per_chunk
        )
        evidence = [
            {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "section": chunk.section,
                "page": chunk.page,
                "text": chunk.text[:char_limit],
            }
            for chunk in chunks
        ]

        evidence_json = json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
Determine whether the academic claim is supported by the
supplied evidence.

Claim:
{claim.text}

Evidence:
{evidence_json}

Check every material part of the claim, including:
- numbers and metric values;
- greater-than and less-than comparisons;
- increases and decreases;
- positive and negative statements;
- methods and experimental conditions;
- population, dataset and scope limitations;
- causal statements;
- whether information is absent from the evidence.

Status definitions:
- supported: all material parts are directly supported.
- partially_supported: some parts are supported, but at
  least one material part is missing or unsupported.
- contradicted: at least one material part is directly
  opposed by the evidence.
- insufficient_evidence: the evidence is relevant but
  cannot establish the claim.

The support_score measures evidence support, not the model's
confidence:
- supported: 0.85 to 1.00
- partially_supported: 0.50 to 0.84
- insufficient_evidence: 0.16 to 0.49
- contradicted: 0.00 to 0.15

Return JSON only:
{{
  "status": "supported",
  "explanation": "Brief evidence-based explanation",
  "support_score": 0.95
}}
""".strip()

    @staticmethod
    def _parse_response(
        raw_response: str,
    ) -> dict[str, Any]:
        text = raw_response.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            result = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CitationValidatorError(
                "Citation validation returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise CitationValidatorError(
                "Citation validation response must be "
                "a JSON object."
            )

        return result

    @staticmethod
    def _create_citation_id(
        claim_id: str,
        evidence_chunk_ids: list[str],
    ) -> str:
        raw_value = "\0".join(
            [
                claim_id,
                *evidence_chunk_ids,
            ]
        )

        digest = hashlib.sha256(
            raw_value.encode("utf-8")
        ).hexdigest()[:16]

        return f"citation-{digest}"

    def _not_found_result(
        self,
        claim: AcademicClaim,
        evidence_chunk_ids: list[str],
        explanation: str,
    ) -> CitationValidation:
        return CitationValidation(
            citation_id=self._create_citation_id(
                claim_id=claim.claim_id,
                evidence_chunk_ids=evidence_chunk_ids,
            ),
            claim_id=claim.claim_id,
            paper_id=claim.paper_id,
            evidence_chunk_ids=evidence_chunk_ids,
            status="not_found",
            explanation=explanation,
            support_score=0.0,
        )