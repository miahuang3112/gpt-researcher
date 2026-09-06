from collections.abc import Sequence

from gpt_researcher.academic.schemas import (
    EvidenceChunk,
    EvidenceMetrics,
)

def calculate_evidence_metrics(
    selected_chunks: Sequence[EvidenceChunk],
    cited_chunk_ids: Sequence[str],
    *,
    claim_support_results: Sequence[bool] | None = None,
    unresolved_contradiction_count: int = 0,
) -> EvidenceMetrics:
    """
    根据实际Evidence和引用结果计算可解释指标。

    不使用模型返回的confidence，也不进行人为加权。
    """

    if unresolved_contradiction_count < 0:
        raise ValueError(
            "unresolved_contradiction_count cannot be negative"
        )

    selected_by_id: dict[str, EvidenceChunk] = {}

    for chunk in selected_chunks:
        if chunk.chunk_id in selected_by_id:
            raise ValueError(
                f"Duplicate selected chunk ID: {chunk.chunk_id}"
            )

        selected_by_id[chunk.chunk_id] = chunk

    cited_id_list = list(cited_chunk_ids)
    unique_cited_ids = set(cited_id_list)

    duplicate_citation_count = (
        len(cited_id_list) - len(unique_cited_ids)
    )

    selected_ids = set(selected_by_id)

    valid_cited_ids = unique_cited_ids & selected_ids
    invalid_cited_ids = unique_cited_ids - selected_ids

    if unique_cited_ids:
        citation_validity = (
            len(valid_cited_ids) / len(unique_cited_ids)
        )
    else:
        citation_validity = 0.0

    if selected_ids:
        citation_coverage = (
            len(valid_cited_ids) / len(selected_ids)
        )
    else:
        citation_coverage = 0.0

    selected_sections = {
        chunk.section.strip()
        for chunk in selected_chunks
        if chunk.section and chunk.section.strip()
    }

    cited_sections = {
        selected_by_id[chunk_id].section.strip()
        for chunk_id in valid_cited_ids
        if selected_by_id[chunk_id].section
        and selected_by_id[chunk_id].section.strip()
    }

    if selected_sections:
        section_coverage = (
            len(cited_sections) / len(selected_sections)
        )
    else:
        section_coverage = 0.0

    support_results = list(claim_support_results or [])

    evaluated_claim_count = len(support_results)
    supported_claim_count = sum(support_results)

    if support_results:
        claim_support_rate: float | None = (
            supported_claim_count / evaluated_claim_count
        )
    else:
        claim_support_rate = None

    return EvidenceMetrics(
        citation_validity=round(citation_validity, 4),
        citation_coverage=round(citation_coverage, 4),
        section_coverage=round(section_coverage, 4),
        claim_support_rate=(
            round(claim_support_rate, 4)
            if claim_support_rate is not None
            else None
        ),
        selected_chunk_count=len(selected_ids),
        cited_chunk_count=len(valid_cited_ids),
        invalid_citation_count=len(invalid_cited_ids),
        duplicate_citation_count=duplicate_citation_count,
        selected_section_count=len(selected_sections),
        cited_section_count=len(cited_sections),
        evaluated_claim_count=evaluated_claim_count,
        supported_claim_count=supported_claim_count,
        unresolved_contradiction_count=(
            unresolved_contradiction_count
        ),
    )