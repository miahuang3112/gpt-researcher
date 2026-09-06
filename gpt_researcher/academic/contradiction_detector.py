from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import ValidationError

from gpt_researcher.academic.schemas import (
    AcademicClaim,
    ContradictionPair,
)


class ContradictionDetectionClient(Protocol):
    """
    矛盾关系判断模型客户端接口。
    """

    async def generate(self, prompt: str) -> str:
        ...


class ContradictionDetectorError(RuntimeError):
    """
    矛盾检测无法完成。
    """


class ContradictionDetector:
    """
    判断来自不同论文的两条Claim之间是什么关系。
    """

    def __init__(
        self,
        client: ContradictionDetectionClient,
    ) -> None:
        self.client = client

    async def compare_claims(
        self,
        claim_a: AcademicClaim,
        claim_b: AcademicClaim,
    ) -> ContradictionPair:
        if claim_a.claim_id == claim_b.claim_id:
            raise ValueError(
                "A claim cannot be compared with itself."
            )

        if claim_a.paper_id == claim_b.paper_id:
            raise ValueError(
                "Cross-paper contradiction detection "
                "requires claims from different papers."
            )

        if self._normalise_text(
            claim_a.text
        ) == self._normalise_text(claim_b.text):
            return ContradictionPair(
                contradiction_id=(
                    self._create_contradiction_id(
                        claim_a.claim_id,
                        claim_b.claim_id,
                    )
                ),
                claim_a_id=claim_a.claim_id,
                claim_b_id=claim_b.claim_id,
                paper_a_id=claim_a.paper_id,
                paper_b_id=claim_b.paper_id,
                relation="consistent",
                condition_differences=[],
                explanation=(
                    "The two claims contain the same "
                    "normalised statement."
                ),
                confidence=1.0,
            )

        prompt = self._build_prompt(
            claim_a=claim_a,
            claim_b=claim_b,
        )

        raw_response = await self.client.generate(prompt)
        result = self._parse_response(raw_response)

        required_fields = {
            "relation",
            "condition_differences",
            "explanation",
            "confidence",
        }

        missing_fields = required_fields.difference(result)

        if missing_fields:
            missing_text = ", ".join(
                sorted(missing_fields)
            )

            raise ContradictionDetectorError(
                "Contradiction response is missing "
                f"required fields: {missing_text}."
            )

        try:
            return ContradictionPair(
                contradiction_id=(
                    self._create_contradiction_id(
                        claim_a.claim_id,
                        claim_b.claim_id,
                    )
                ),
                claim_a_id=claim_a.claim_id,
                claim_b_id=claim_b.claim_id,
                paper_a_id=claim_a.paper_id,
                paper_b_id=claim_b.paper_id,
                relation=result["relation"],
                condition_differences=result[
                    "condition_differences"
                ],
                explanation=result["explanation"],
                confidence=result["confidence"],
            )
        except ValidationError as exc:
            raise ContradictionDetectorError(
                "Contradiction response failed "
                "schema validation."
            ) from exc

    @staticmethod
    def _build_prompt(
        claim_a: AcademicClaim,
        claim_b: AcademicClaim,
    ) -> str:
        claim_data = {
            "claim_a": {
                "claim_id": claim_a.claim_id,
                "paper_id": claim_a.paper_id,
                "claim_type": claim_a.claim_type,
                "text": claim_a.text,
            },
            "claim_b": {
                "claim_id": claim_b.claim_id,
                "paper_id": claim_b.paper_id,
                "claim_type": claim_b.claim_type,
                "text": claim_b.text,
            },
        }

        claim_json = json.dumps(
            claim_data,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
Determine the relationship between two academic claims.

Claims:
{claim_json}

Relation definitions:
- contradiction: the claims address the same question under
  sufficiently comparable conditions but reach incompatible
  conclusions.
- partial_contradiction: the claims overlap and conflict on
  at least one material part, while other parts remain
  compatible.
- different_conditions: the results appear different, but
  the difference can reasonably be explained by different
  populations, datasets, metrics, methods, interventions,
  baselines, time periods, sample sizes, or evaluation
  settings.
- consistent: the claims support the same conclusion or
  direction.
- unrelated: the claims do not address the same proposition.
- uncertain: the supplied claims do not contain enough
  information to determine the relationship.

Rules:
1. Do not label different numerical values as contradiction
   unless the metric and experimental conditions are
   comparable.
2. Do not treat different datasets or populations as direct
   contradiction.
3. Check negation, greater-than and less-than relations,
   increases, decreases, and causal language.
4. Record every identifiable condition difference.
5. Ignore the claims' original confidence scores.
6. Do not invent experimental conditions.
7. A condition difference may only be reported when it is
   explicitly stated in at least one of the supplied claims.
8. Different paper IDs alone are not evidence of different
   methods, implementations, training procedures, or model
   architectures.
9. Never hypothesise an unstated condition difference. If
   the claims state that the dataset, metric, protocol and
   evaluated implementations are the same but report
   incompatible outcomes, classify the relation as
   contradiction.

Allowed condition differences:
- population
- dataset
- metric
- method
- intervention
- baseline
- time_period
- sample_size
- evaluation_setting
- other

Return JSON only:
{{
  "relation": "contradiction",
  "condition_differences": [],
  "explanation": "Brief comparison grounded in both claims",
  "confidence": 0.90
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
            raise ContradictionDetectorError(
                "Contradiction detection returned "
                "invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ContradictionDetectorError(
                "Contradiction response must be "
                "a JSON object."
            )

        return result

    @staticmethod
    def _normalise_text(text: str) -> str:
        return " ".join(
            text.lower().split()
        )

    @staticmethod
    def _create_contradiction_id(
        claim_a_id: str,
        claim_b_id: str,
    ) -> str:
        ordered_ids = sorted(
            [
                claim_a_id,
                claim_b_id,
            ]
        )

        raw_value = "\0".join(ordered_ids)

        digest = hashlib.sha256(
            raw_value.encode("utf-8")
        ).hexdigest()[:16]

        return f"contradiction-{digest}"