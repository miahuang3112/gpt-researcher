from collections.abc import Sequence

from gpt_researcher.academic.citation_validator import (
    CitationValidator,
)
from gpt_researcher.academic.evaluation_claims import (
    build_evaluation_claims,
)
from gpt_researcher.academic.schemas import (
    CitationValidation,
    DimensionEvaluation,
    EvidenceChunk,
)


class EvaluationCitationVerifierError(RuntimeError):
    """
    论文评分引用验证过程中发生的数据一致性错误。
    """


async def verify_evaluation_claims(
    *,
    validator: CitationValidator,
    paper_id: str,
    evaluation: DimensionEvaluation,
    evidence_chunks: Sequence[EvidenceChunk],
    max_claims: int = 4,
) -> list[CitationValidation]:
    """
    验证一个评分维度中的主要评价陈述。

    执行过程：
    1. 把DimensionEvaluation转换为AcademicClaim；
    2. 根据evidence_chunk_ids找到对应EvidenceChunk；
    3. 使用CitationValidator逐条验证；
    4. 返回每条Claim的CitationValidation。
    """
    claims = build_evaluation_claims(
        paper_id=paper_id,
        evaluation=evaluation,
        max_claims=max_claims,
    )

    evidence_by_id = {
        chunk.chunk_id: chunk
        for chunk in evidence_chunks
    }

    validations: list[CitationValidation] = []

    for claim in claims:
        claim_evidence = [
            evidence_by_id[chunk_id]
            for chunk_id in claim.evidence_chunk_ids
            if chunk_id in evidence_by_id
        ]

        validation = await validator.validate_claim(
            claim,
            claim_evidence,
        )

        if validation.claim_id != claim.claim_id:
            raise EvaluationCitationVerifierError(
                "CitationValidator returned an unexpected claim_id: "
                f"expected {claim.claim_id!r}, "
                f"received {validation.claim_id!r}"
            )

        if validation.paper_id != paper_id:
            raise EvaluationCitationVerifierError(
                "CitationValidator returned an unexpected paper_id: "
                f"expected {paper_id!r}, "
                f"received {validation.paper_id!r}"
            )

        validations.append(validation)

    return validations


def build_claim_support_results(
    validations: Sequence[CitationValidation],
) -> list[bool]:
    """
    将引用验证状态转换为Evidence Metrics使用的布尔结果。

    只有supported算作通过。
    partially_supported、contradicted、insufficient_evidence
    和not_found都算作没有完全通过。
    """
    return [
        validation.status == "supported"
        for validation in validations
    ]