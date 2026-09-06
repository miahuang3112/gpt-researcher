import hashlib

from gpt_researcher.academic.schemas import (
    AcademicClaim,
    DimensionEvaluation,
)


def build_evaluation_claims(
    paper_id: str,
    evaluation: DimensionEvaluation,
    *,
    max_claims: int = 4,
) -> list[AcademicClaim]:
    """
    将一个评分维度的评价结果转换为可验证的学术主张。

    验证顺序：
    1. 评分理由 rationale
    2. 问题 concerns
    3. 优点 strengths

    max_claims 用来限制每个维度调用 CitationValidator 的次数，
    避免九个维度产生过多 Ollama 请求。
    """
    normalized_paper_id = paper_id.strip()

    if not normalized_paper_id:
        raise ValueError("paper_id cannot be empty")

    if max_claims <= 0:
        raise ValueError("max_claims must be greater than 0")

    candidates: list[tuple[str, str]] = [
        ("other", evaluation.rationale),
    ]

    candidates.extend(
        ("limitation", concern)
        for concern in evaluation.concerns
    )

    candidates.extend(
        ("other", strength)
        for strength in evaluation.strengths
    )

    evidence_chunk_ids = list(
        dict.fromkeys(evaluation.evidence_chunk_ids)
    )

    claims: list[AcademicClaim] = []
    seen_texts: set[str] = set()

    for claim_type, raw_text in candidates:
        text = " ".join(raw_text.split())

        if not text:
            continue

        normalized_text = text.casefold()

        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)

        claim_id = _create_claim_id(
            paper_id=normalized_paper_id,
            dimension=evaluation.dimension,
            claim_type=claim_type,
            text=text,
        )

        claims.append(
            AcademicClaim(
                claim_id=claim_id,
                paper_id=normalized_paper_id,
                text=text,
                claim_type=claim_type,
                evidence_chunk_ids=evidence_chunk_ids,
                confidence=0.0,
            )
        )

        if len(claims) >= max_claims:
            break

    return claims


def _create_claim_id(
    *,
    paper_id: str,
    dimension: str,
    claim_type: str,
    text: str,
) -> str:
    """
    根据论文、评分维度和主张内容生成稳定且可重复的 ID。
    """
    value = "\n".join(
        [
            paper_id,
            dimension,
            claim_type,
            text,
        ]
    )

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]

    return f"evaluation-claim-{digest}"