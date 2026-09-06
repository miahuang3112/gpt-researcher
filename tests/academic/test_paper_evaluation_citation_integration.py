import json

import pytest

from gpt_researcher.academic.paper_evaluation_pipeline import (
    REQUIRED_DIMENSIONS,
    PaperEvaluationPipeline,
)
from gpt_researcher.academic.schemas import (
    CitationValidation,
    EvidenceChunk,
)

from gpt_researcher.academic.citation_validator import (
    CitationValidatorError,
)


class FakeDimensionEvaluationClient:
    """
    按照九个维度的执行顺序返回合法评分结果。
    """

    def __init__(self):
        self.call_index = 0

    async def generate(self, prompt: str) -> str:
        dimension = REQUIRED_DIMENSIONS[self.call_index]
        self.call_index += 1

        dimension_value = getattr(
            dimension,
            "value",
            dimension,
        )

        return json.dumps(
            {
                "dimension": dimension_value,
                "status": "scored",
                "score": 4,
                "rationale": (
                    "The available evidence supports "
                    "this evaluation."
                ),
                "evidence_chunk_ids": ["chunk-1"],
                "strengths": [
                    "The paper provides clear evidence."
                ],
                "concerns": [],
            }
        )


class FakeCitationValidator:
    def __init__(self):
        self.calls = []

    async def validate_claim(
        self,
        claim,
        evidence_chunks,
    ):
        chunks = list(evidence_chunks)

        self.calls.append(
            {
                "claim": claim,
                "evidence_chunks": chunks,
            }
        )

        return CitationValidation(
            citation_id=f"citation-{claim.claim_id}",
            claim_id=claim.claim_id,
            paper_id=claim.paper_id,
            evidence_chunk_ids=[
                chunk.chunk_id
                for chunk in chunks
            ],
            status="supported",
            explanation=(
                "The selected evidence supports the claim."
            ),
            support_score=1.0,
        )

class FailingCitationValidator:
    def __init__(self):
        self.call_count = 0

    async def validate_claim(
        self,
        claim,
        evidence_chunks,
    ):
        self.call_count += 1

        raise CitationValidatorError(
            "Citation validation returned invalid JSON."
        )

class RecordingEvaluator:
    """
    记录Pipeline传给最终聚合器的维度结果。
    """

    def __init__(self):
        self.dimension_evaluations = []

    def evaluate(
        self,
        *,
        paper_id,
        evidence_chunks,
        dimension_evaluations,
    ):
        self.dimension_evaluations = list(
            dimension_evaluations
        )

        return self.dimension_evaluations


def make_evidence() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id="chunk-1",
            paper_id="paper-1",
            text=(
                "The paper describes its research method, "
                "experimental design, results, limitations, "
                "baselines, and implementation details."
            ),
            section="Methods",
            page=1,
        )
    ]


@pytest.mark.asyncio
async def test_pipeline_calculates_claim_support_rate():
    dimension_client = FakeDimensionEvaluationClient()
    citation_validator = FakeCitationValidator()
    evaluator = RecordingEvaluator()

    pipeline = PaperEvaluationPipeline(
        client=dimension_client,
        evaluator=evaluator,
        citation_validator=citation_validator,
        max_chunks_per_dimension=1,
        max_claims_per_dimension=2,
    )

    await pipeline.evaluate(
        paper_id="paper-1",
        evidence_chunks=make_evidence(),
    )

    assert citation_validator.calls

    assert len(evaluator.dimension_evaluations) == len(
        REQUIRED_DIMENSIONS
    )

    for result in evaluator.dimension_evaluations:
        assert result.verification is not None
        assert (
            result.verification.metrics.claim_support_rate
            == 1.0
        )


@pytest.mark.asyncio
async def test_pipeline_without_validator_keeps_rate_none():
    dimension_client = FakeDimensionEvaluationClient()
    evaluator = RecordingEvaluator()

    pipeline = PaperEvaluationPipeline(
        client=dimension_client,
        evaluator=evaluator,
        citation_validator=None,
        max_chunks_per_dimension=1,
    )

    await pipeline.evaluate(
        paper_id="paper-1",
        evidence_chunks=make_evidence(),
    )

    for result in evaluator.dimension_evaluations:
        assert result.verification is not None
        assert (
            result.verification.metrics.claim_support_rate
            is None
        )


@pytest.mark.asyncio
async def test_empty_evidence_skips_citation_validator():
    dimension_client = FakeDimensionEvaluationClient()
    citation_validator = FakeCitationValidator()
    evaluator = RecordingEvaluator()

    pipeline = PaperEvaluationPipeline(
        client=dimension_client,
        evaluator=evaluator,
        citation_validator=citation_validator,
    )

    await pipeline.evaluate(
        paper_id="paper-1",
        evidence_chunks=[],
    )

    assert citation_validator.calls == []
    assert dimension_client.call_index == 0

    for result in evaluator.dimension_evaluations:
        assert result.status == "insufficient_evidence"
        assert result.verification is not None
        assert (
            result.verification.status
            == "insufficient_evidence"
        )

@pytest.mark.asyncio
async def test_pipeline_continues_when_citation_validation_fails():
    dimension_client = FakeDimensionEvaluationClient()
    citation_validator = FailingCitationValidator()
    evaluator = RecordingEvaluator()

    pipeline = PaperEvaluationPipeline(
        client=dimension_client,
        evaluator=evaluator,
        citation_validator=citation_validator,
        max_chunks_per_dimension=1,
        max_claims_per_dimension=2,
    )

    await pipeline.evaluate(
        paper_id="paper-1",
        evidence_chunks=make_evidence(),
    )

    assert citation_validator.call_count == len(
        REQUIRED_DIMENSIONS
    )

    assert len(
        evaluator.dimension_evaluations
    ) == len(REQUIRED_DIMENSIONS)

    for result in evaluator.dimension_evaluations:
        assert result.verification is not None
        assert (
            result.verification.metrics.claim_support_rate
            is None
        )
        assert (
            result.verification.status
            == "insufficient_evidence"
        )

@pytest.mark.asyncio
async def test_pipeline_resumes_completed_dimensions():
    first_evaluator = RecordingEvaluator()

    first_pipeline = PaperEvaluationPipeline(
        client=FakeDimensionEvaluationClient(),
        evaluator=first_evaluator,
        max_chunks_per_dimension=1,
    )

    await first_pipeline.evaluate(
        paper_id="paper-1",
        evidence_chunks=make_evidence(),
    )

    completed_results = (
        first_evaluator.dimension_evaluations[:1]
    )
    checkpoint_snapshots = []

    def record_checkpoint(results):
        checkpoint_snapshots.append(
            list(results)
        )

    resumed_evaluator = RecordingEvaluator()
    resumed_client = FakeDimensionEvaluationClient()

    # 模拟第一维已经在之前的运行中消耗。
    await resumed_client.generate("")

    resumed_pipeline = PaperEvaluationPipeline(
        client=resumed_client,
        evaluator=resumed_evaluator,
        max_chunks_per_dimension=1,
    )

    await resumed_pipeline.evaluate(
        paper_id="paper-1",
        evidence_chunks=make_evidence(),
        completed_dimension_results=(
            completed_results
        ),
        on_dimension_complete=record_checkpoint,
    )

    assert len(checkpoint_snapshots) == (
        len(REQUIRED_DIMENSIONS) - 1
    )

    assert len(checkpoint_snapshots[0]) == 2

    assert len(checkpoint_snapshots[-1]) == len(
        REQUIRED_DIMENSIONS
    )

    assert len(
        resumed_evaluator.dimension_evaluations
    ) == len(REQUIRED_DIMENSIONS)

    assert (
        resumed_evaluator.dimension_evaluations[0]
        == completed_results[0]
    )