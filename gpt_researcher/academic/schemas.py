from typing import Literal

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """
    保存一篇论文的标准化元数据。
    不同学术搜索平台返回的数据，最终都转换成这个统一格式。
    """

    paper_id: str
    title: str
    source_ids: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    url: str | None = None
    abstract: str | None = None
    citation_count: int | None = Field(
        default=None,
        ge=0,
    )
    is_retracted: bool | None = None
    is_open_access: bool | None = None
    full_text_url: str | None = None

    source_type: Literal[
        "journal",
        "conference",
        "preprint",
        "book",
        "report",
        "web",
        "unknown",
    ] = "unknown"

    retrieved_from: Literal[
        "openalex",
        "crossref",
        "semantic_scholar",
        "arxiv",
        "web",
        "uploaded",
        "unknown",
    ] = "unknown"


class EvidenceChunk(BaseModel):
    """
    保存从论文或网页中抽取的一段证据，以及它的检索分数。
    """

    chunk_id: str
    paper_id: str
    text: str

    section: str | None = None
    page: int | None = None
    source_url: str | None = None
    content_warnings: list[
        Literal["prompt_injection_pattern"]
    ] = Field(default_factory=list)

    bm25_score: float | None = None
    vector_score: float | None = None
    fused_score: float | None = None

class AcademicClaim(BaseModel):
    """
    从论文或外部来源中抽取的一条结构化主张。
    """

    claim_id: str
    paper_id: str
    text: str

    claim_type: Literal[
        "method",
        "result",
        "comparison",
        "limitation",
        "background",
        "other",
    ] = "other"

    evidence_chunk_ids: list[str] = Field(default_factory=list)

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

class ContradictionPair(BaseModel):
    """
    保存两条学术Claim之间的关系和矛盾判断。
    """

    contradiction_id: str

    claim_a_id: str
    claim_b_id: str

    paper_a_id: str
    paper_b_id: str

    relation: Literal[
        "contradiction",
        "partial_contradiction",
        "different_conditions",
        "consistent",
        "unrelated",
        "uncertain",
    ]

    condition_differences: list[
        Literal[
            "population",
            "dataset",
            "metric",
            "method",
            "intervention",
            "baseline",
            "time_period",
            "sample_size",
            "evaluation_setting",
            "other",
        ]
    ] = Field(default_factory=list)

    explanation: str

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class ContradictionWarning(BaseModel):
    """A review warning linked to both claims and their evidence."""

    evaluation_dimension: "EvaluationDimension"
    pair: ContradictionPair
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    review_required: bool = True


class AnalysisWarning(BaseModel):
    """A non-fatal analysis problem that must remain visible to callers."""

    code: Literal[
        "contradiction_detection_not_configured",
        "contradiction_detection_unavailable",
    ]
    message: str
    evaluation_dimension: "EvaluationDimension | None" = None
    error_type: str | None = None


class CitationValidation(BaseModel):
    """
    保存一条Claim和对应证据之间的引用验证结果。
    """

    citation_id: str
    claim_id: str
    paper_id: str

    evidence_chunk_ids: list[str] = Field(
        default_factory=list
    )

    status: Literal[
        "supported",
        "partially_supported",
        "contradicted",
        "insufficient_evidence",
        "not_found",
    ]

    explanation: str

    support_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
        

class AcademicSearchResult(BaseModel):
    """
    保存学术检索结果以及fallback执行情况。
    """

    query: str
    papers: list[PaperMetadata] = Field(
        default_factory=list
    )

    provider_used: Literal[
        "openalex",
        "crossref",
    ]

    fallback_used: bool = False
    errors: list[str] = Field(
        default_factory=list
    )

class SourceQualityScore(BaseModel):
    """
    保存一篇来源的学术质量评分及评分依据。
    """

    paper_id: str

    total_score: float = Field(
        ge=0,
        le=100,
    )

    grade: Literal[
        "high",
        "medium",
        "low",
        "rejected",
    ]

    source_type_score: float = Field(
        ge=0,
        le=30,
    )
    traceability_score: float = Field(
        ge=0,
        le=25,
    )
    evidence_access_score: float = Field(
        ge=0,
        le=20,
    )
    metadata_completeness_score: float = Field(
        ge=0,
        le=15,
    )
    citation_score: float = Field(
        ge=0,
        le=10,
    )

    score_cap: float = Field(
        ge=0,
        le=100,
    )

    reasons: list[str] = Field(default_factory=list)

EvaluationDimension = Literal[
    "research_question_clarity",
    "methodological_rigor",
    "experimental_design",
    "baseline_quality",
    "result_validity",
    "reproducibility",
    "novelty",
    "limitations_discussion",
    "citation_quality",
]


EvaluationStatus = Literal[
    "scored",
    "insufficient_evidence",
]


class ExternalScholarlyRecord(BaseModel):
    """A deduplicated external metadata record used as bounded context."""

    external_evidence_id: str
    paper: PaperMetadata
    eligible_for_novelty: bool = False
    temporal_note: str


class ExternalEvidenceSummary(BaseModel):
    """Auditable outcome of optional external scholarly retrieval."""

    status: Literal[
        "not_requested",
        "available",
        "no_results",
        "unavailable",
    ]
    query: str | None = None
    provider_used: Literal["openalex", "crossref"] | None = None
    fallback_used: bool = False
    target_year: int | None = None
    records: list[ExternalScholarlyRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


OverallAssessment = Literal[
    "strong",
    "solid",
    "mixed",
    "weak",
    "insufficient_evidence",
]

VerificationStatus = Literal[
    "checks_passed",
    "review_required",
    "insufficient_evidence",
]


class EvidenceMetrics(BaseModel):
    """
    可直接观察和复现的Evidence指标。
    不代表评价正确的概率。
    """

    citation_validity: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    section_coverage: float = Field(ge=0.0, le=1.0)

    claim_support_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    selected_chunk_count: int = Field(ge=0)
    cited_chunk_count: int = Field(ge=0)
    invalid_citation_count: int = Field(ge=0)
    duplicate_citation_count: int = Field(ge=0)

    selected_section_count: int = Field(ge=0)
    cited_section_count: int = Field(ge=0)

    evaluated_claim_count: int = Field(ge=0)
    supported_claim_count: int = Field(ge=0)

    unresolved_contradiction_count: int = Field(ge=0)


class VerificationResult(BaseModel):
    """
    自动证据验证结果。
    checks_passed不表示已经被人工确认。
    """

    status: VerificationStatus

    reasons: list[str] = Field(
        default_factory=list,
    )

    metrics: EvidenceMetrics

class DimensionEvaluation(BaseModel):
    """论文单个评价维度的结构化结果。"""

    dimension: EvaluationDimension

    status: EvaluationStatus = "scored"

    score: float | None = Field(
        default=None,
        ge=1.0,
        le=5.0,
    )

    rationale: str

    evidence_chunk_ids: list[str] = Field(
        default_factory=list
    )

    external_evidence_ids: list[str] = Field(
        default_factory=list
    )

    strengths: list[str] = Field(
        default_factory=list
    )

    concerns: list[str] = Field(
        default_factory=list
    )

    verification: VerificationResult | None = None

class PaperEvaluation(BaseModel):
    """一篇论文的最终多维评价结果。"""

    paper_id: str

    dimensions: list[DimensionEvaluation]

    overall_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    evidence_coverage: float = Field(
        ge=0.0,
        le=1.0,
    )

    assessment: OverallAssessment

    missing_dimensions: list[EvaluationDimension] = Field(
        default_factory=list
    )

    strengths: list[str] = Field(
        default_factory=list
    )

    major_concerns: list[str] = Field(
        default_factory=list
    )

    contradiction_warnings: list[ContradictionWarning] = Field(
        default_factory=list
    )

    analysis_warnings: list[AnalysisWarning] = Field(
        default_factory=list
    )

    external_evidence: ExternalEvidenceSummary | None = None
