from __future__ import annotations

from collections.abc import Mapping
from math import isclose

from gpt_researcher.academic.schemas import (
    DimensionEvaluation,
    EvaluationDimension,
    EvidenceChunk,
    OverallAssessment,
    PaperEvaluation,
)


REQUIRED_DIMENSIONS: tuple[EvaluationDimension, ...] = (
    "research_question_clarity",
    "methodological_rigor",
    "experimental_design",
    "baseline_quality",
    "result_validity",
    "reproducibility",
    "novelty",
    "limitations_discussion",
    "citation_quality",
)


DEFAULT_DIMENSION_WEIGHTS: dict[
    EvaluationDimension,
    float,
] = {
    "research_question_clarity": 0.10,
    "methodological_rigor": 0.20,
    "experimental_design": 0.15,
    "baseline_quality": 0.10,
    "result_validity": 0.15,
    "reproducibility": 0.10,
    "novelty": 0.10,
    "limitations_discussion": 0.05,
    "citation_quality": 0.05,
}


DIMENSION_DISPLAY_NAMES: dict[
    EvaluationDimension,
    str,
] = {
    "research_question_clarity": "Research question clarity",
    "methodological_rigor": "Methodological rigor",
    "experimental_design": "Experimental design",
    "baseline_quality": "Baseline quality",
    "result_validity": "Result validity",
    "reproducibility": "Reproducibility",
    "novelty": "Novelty",
    "limitations_discussion": "Limitations discussion",
    "citation_quality": "Citation quality",
}


class PaperEvaluatorError(ValueError):
    """论文评分输入不完整或不一致。"""


class PaperEvaluator:
    """
    验证各维度评分，并计算论文总分。

    该类本身不调用LLM，只处理结构化评分结果。
    """

    def __init__(
        self,
        dimension_weights: Mapping[str, float] | None = None,
        minimum_evidence_coverage: float = 0.80,
    ) -> None:
        if not 0.0 <= minimum_evidence_coverage <= 1.0:
            raise PaperEvaluatorError(
                "minimum_evidence_coverage must be "
                "between 0 and 1."
            )

        raw_weights = (
            dict(dimension_weights)
            if dimension_weights is not None
            else dict(DEFAULT_DIMENSION_WEIGHTS)
        )

        required_set = set(REQUIRED_DIMENSIONS)
        supplied_set = set(raw_weights)

        if supplied_set != required_set:
            missing = sorted(required_set - supplied_set)
            extra = sorted(supplied_set - required_set)

            raise PaperEvaluatorError(
                "Dimension weights do not match required "
                f"dimensions. Missing={missing}, extra={extra}"
            )

        if any(weight <= 0 for weight in raw_weights.values()):
            raise PaperEvaluatorError(
                "Every dimension weight must be greater than 0."
            )

        if not isclose(
            sum(raw_weights.values()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PaperEvaluatorError(
                "Dimension weights must sum to 1.0."
            )

        self.dimension_weights = {
            dimension: float(raw_weights[dimension])
            for dimension in REQUIRED_DIMENSIONS
        }

        self.minimum_evidence_coverage = (
            minimum_evidence_coverage
        )

    def evaluate(
        self,
        paper_id: str,
        evidence_chunks: list[EvidenceChunk],
        dimension_evaluations: list[DimensionEvaluation],
    ) -> PaperEvaluation:
        normalized_paper_id = paper_id.strip()

        if not normalized_paper_id:
            raise PaperEvaluatorError(
                "paper_id cannot be empty."
            )

        evidence_by_id = self._index_evidence(
            evidence_chunks=evidence_chunks,
            paper_id=normalized_paper_id,
        )

        evaluations_by_dimension = (
            self._validate_dimension_evaluations(
                dimension_evaluations=dimension_evaluations,
                evidence_by_id=evidence_by_id,
            )
        )

        ordered_evaluations = [
            evaluations_by_dimension[dimension]
            for dimension in REQUIRED_DIMENSIONS
        ]

        scored_evaluations = [
            evaluation
            for evaluation in ordered_evaluations
            if evaluation.status == "scored"
        ]

        evidence_coverage = sum(
            self.dimension_weights[evaluation.dimension]
            for evaluation in scored_evaluations
        )

        overall_score = self._calculate_overall_score(
            scored_evaluations,
            evidence_coverage,
        )

        assessment = self._create_assessment(
            overall_score
        )

        missing_dimensions = [
            evaluation.dimension
            for evaluation in ordered_evaluations
            if evaluation.status == "insufficient_evidence"
        ]

        strengths = self._collect_strengths(
            ordered_evaluations
        )

        major_concerns = self._collect_concerns(
            ordered_evaluations
        )

        return PaperEvaluation(
            paper_id=normalized_paper_id,
            dimensions=ordered_evaluations,
            overall_score=overall_score,
            evidence_coverage=round(
                evidence_coverage,
                4,
            ),
            assessment=assessment,
            missing_dimensions=missing_dimensions,
            strengths=strengths,
            major_concerns=major_concerns,
        )

    @staticmethod
    def _index_evidence(
        evidence_chunks: list[EvidenceChunk],
        paper_id: str,
    ) -> dict[str, EvidenceChunk]:
        evidence_by_id: dict[str, EvidenceChunk] = {}

        for chunk in evidence_chunks:
            if chunk.paper_id != paper_id:
                raise PaperEvaluatorError(
                    f"Evidence chunk {chunk.chunk_id} belongs "
                    f"to paper {chunk.paper_id}, not {paper_id}."
                )

            if chunk.chunk_id in evidence_by_id:
                raise PaperEvaluatorError(
                    "Duplicate evidence chunk ID: "
                    f"{chunk.chunk_id}"
                )

            evidence_by_id[chunk.chunk_id] = chunk

        return evidence_by_id

    def _validate_dimension_evaluations(
        self,
        dimension_evaluations: list[DimensionEvaluation],
        evidence_by_id: dict[str, EvidenceChunk],
    ) -> dict[
        EvaluationDimension,
        DimensionEvaluation,
    ]:
        evaluations_by_dimension: dict[
            EvaluationDimension,
            DimensionEvaluation,
        ] = {}

        for evaluation in dimension_evaluations:
            dimension = evaluation.dimension

            if dimension in evaluations_by_dimension:
                raise PaperEvaluatorError(
                    "Duplicate evaluation dimension: "
                    f"{dimension}"
                )

            self._validate_single_evaluation(
                evaluation=evaluation,
                evidence_by_id=evidence_by_id,
            )

            evaluations_by_dimension[dimension] = (
                evaluation
            )

        missing_dimensions = [
            dimension
            for dimension in REQUIRED_DIMENSIONS
            if dimension not in evaluations_by_dimension
        ]

        if missing_dimensions:
            raise PaperEvaluatorError(
                "Missing evaluation dimensions: "
                f"{missing_dimensions}"
            )

        return evaluations_by_dimension

    @staticmethod
    def _validate_single_evaluation(
        evaluation: DimensionEvaluation,
        evidence_by_id: dict[str, EvidenceChunk],
    ) -> None:
        if not evaluation.rationale.strip():
            raise PaperEvaluatorError(
                f"{evaluation.dimension} requires a rationale."
            )

        if len(set(evaluation.evidence_chunk_ids)) != len(
            evaluation.evidence_chunk_ids
        ):
            raise PaperEvaluatorError(
                "Duplicate evidence IDs in dimension: "
                f"{evaluation.dimension}"
            )

        unknown_evidence_ids = [
            chunk_id
            for chunk_id in evaluation.evidence_chunk_ids
            if chunk_id not in evidence_by_id
        ]

        if unknown_evidence_ids:
            raise PaperEvaluatorError(
                "Unknown evidence IDs for "
                f"{evaluation.dimension}: "
                f"{unknown_evidence_ids}"
            )

        if evaluation.status == "scored":
            if evaluation.score is None:
                raise PaperEvaluatorError(
                    f"{evaluation.dimension} is scored "
                    "but has no score."
                )

            if not evaluation.evidence_chunk_ids:
                raise PaperEvaluatorError(
                    f"{evaluation.dimension} is scored "
                    "but cites no evidence."
                )

        if (
            evaluation.status == "insufficient_evidence"
            and evaluation.score is not None
        ):
            raise PaperEvaluatorError(
                f"{evaluation.dimension} has insufficient "
                "evidence and therefore cannot have a score."
            )

    def _calculate_overall_score(
        self,
        evaluations: list[DimensionEvaluation],
        evidence_coverage: float,
    ) -> float | None:
        if (
            evidence_coverage
            < self.minimum_evidence_coverage
        ):
            return None

        if evidence_coverage == 0:
            return None

        weighted_score = sum(
            (
                float(evaluation.score)
                / 5.0
                * 100.0
                * self.dimension_weights[
                    evaluation.dimension
                ]
            )
            for evaluation in evaluations
            if evaluation.score is not None
        )

        normalized_score = (
            weighted_score
            / evidence_coverage
        )

        return round(normalized_score, 2)

    @staticmethod
    def _create_assessment(
        overall_score: float | None,
    ) -> OverallAssessment:
        if overall_score is None:
            return "insufficient_evidence"

        if overall_score >= 85:
            return "strong"

        if overall_score >= 70:
            return "solid"

        if overall_score >= 55:
            return "mixed"

        return "weak"

    @staticmethod
    def _collect_strengths(
        evaluations: list[DimensionEvaluation],
    ) -> list[str]:
        strengths: list[str] = []

        for evaluation in evaluations:
            if (
                evaluation.status == "scored"
                and evaluation.score is not None
                and evaluation.score >= 4.0
            ):
                if evaluation.strengths:
                    strengths.extend(evaluation.strengths)
                else:
                    strengths.append(
                        f"{DIMENSION_DISPLAY_NAMES[evaluation.dimension]}: "
                        f"{evaluation.rationale}"
                    )

        return PaperEvaluator._deduplicate_text(
            strengths
        )

    @staticmethod
    def _collect_concerns(
        evaluations: list[DimensionEvaluation],
    ) -> list[str]:
        concerns: list[str] = []

        for evaluation in evaluations:
            if evaluation.concerns:
                concerns.extend(evaluation.concerns)

            elif evaluation.status == "insufficient_evidence":
                concerns.append(
                    f"{DIMENSION_DISPLAY_NAMES[evaluation.dimension]}: "
                    "insufficient evidence."
                )

            elif (
                evaluation.score is not None
                and evaluation.score <= 2.5
            ):
                concerns.append(
                    f"{DIMENSION_DISPLAY_NAMES[evaluation.dimension]}: "
                    f"{evaluation.rationale}"
                )

        return PaperEvaluator._deduplicate_text(
            concerns
        )

    @staticmethod
    def _deduplicate_text(
        values: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(value.split())

            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)

        return result