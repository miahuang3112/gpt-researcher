from gpt_researcher.academic.schemas import (
    EvidenceMetrics,
    VerificationResult,
)

def determine_verification_status(
    metrics: EvidenceMetrics,
    *,
    minimum_cited_chunks: int = 2,
    minimum_citation_coverage: float = 0.75,
    minimum_claim_support_rate: float = 0.80,
    minimum_cited_sections: int = 2,
) -> VerificationResult:
    """
    根据透明、可配置的规则决定验证状态。

    这些阈值是操作策略，不是正确概率。
    """

    if minimum_cited_chunks <= 0:
        raise ValueError(
            "minimum_cited_chunks must be greater than zero"
        )

    if not 0.0 <= minimum_citation_coverage <= 1.0:
        raise ValueError(
            "minimum_citation_coverage must be between zero and one"
        )

    if not 0.0 <= minimum_claim_support_rate <= 1.0:
        raise ValueError(
            "minimum_claim_support_rate must be between zero and one"
        )

    if minimum_cited_sections <= 0:
        raise ValueError(
            "minimum_cited_sections must be greater than zero"
        )

    insufficient_reasons: list[str] = []

    if metrics.selected_chunk_count == 0:
        insufficient_reasons.append(
            "No evidence chunks were selected."
        )

    if metrics.cited_chunk_count < minimum_cited_chunks:
        insufficient_reasons.append(
            "Too few valid evidence chunks were cited: "
            f"{metrics.cited_chunk_count} found, "
            f"{minimum_cited_chunks} required."
        )

    if insufficient_reasons:
        return VerificationResult(
            status="insufficient_evidence",
            reasons=insufficient_reasons,
            metrics=metrics,
        )

    review_reasons: list[str] = []

    if metrics.invalid_citation_count > 0:
        review_reasons.append(
            "The evaluation contains "
            f"{metrics.invalid_citation_count} invalid citation(s)."
        )

    if metrics.duplicate_citation_count > 0:
        review_reasons.append(
            "The evaluation contains "
            f"{metrics.duplicate_citation_count} duplicate citation(s)."
        )

    if (
        metrics.citation_coverage
        < minimum_citation_coverage
    ):
        review_reasons.append(
            "Citation coverage is below the required threshold: "
            f"{metrics.citation_coverage:.2f} found, "
            f"{minimum_citation_coverage:.2f} required."
        )

    if (
        metrics.cited_section_count
        < minimum_cited_sections
    ):
        review_reasons.append(
            "Evidence does not cover enough sections: "
            f"{metrics.cited_section_count} found, "
            f"{minimum_cited_sections} required."
        )

    if metrics.claim_support_rate is None:
        review_reasons.append(
            "Claim support has not been independently evaluated."
        )
    elif (
        metrics.claim_support_rate
        < minimum_claim_support_rate
    ):
        review_reasons.append(
            "Claim support rate is below the required threshold: "
            f"{metrics.claim_support_rate:.2f} found, "
            f"{minimum_claim_support_rate:.2f} required."
        )

    if metrics.unresolved_contradiction_count > 0:
        review_reasons.append(
            "The evaluation contains "
            f"{metrics.unresolved_contradiction_count} "
            "unresolved contradiction(s)."
        )

    if review_reasons:
        return VerificationResult(
            status="review_required",
            reasons=review_reasons,
            metrics=metrics,
        )

    return VerificationResult(
        status="checks_passed",
        reasons=[
            "All configured automated evidence checks passed."
        ],
        metrics=metrics,
    )