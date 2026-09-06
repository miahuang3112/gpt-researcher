from __future__ import annotations

import json
from pathlib import Path

from gpt_researcher.academic.citation_validator import CitationValidator
from gpt_researcher.academic.ollama_citation_client import (
    OllamaCitationClient,
)
from gpt_researcher.academic.ollama_paper_evaluator_client import (
    OllamaPaperEvaluatorClient,
)
from gpt_researcher.academic.paper_chunker import (
    AcademicPaperChunker,
)
from gpt_researcher.academic.paper_evaluation_pipeline import (
    PaperEvaluationPipeline,
)
from gpt_researcher.academic.external_evidence import (
    ExternalScholarlyEvidenceRetriever,
)
from gpt_researcher.academic.fallback_policy import AcademicSearchFallback
from gpt_researcher.academic.schemas import (
    DimensionEvaluation,
    ExternalEvidenceSummary,
)
from gpt_researcher.academic.vector_retriever import OllamaEmbeddingClient


def write_checkpoint(
    checkpoint_path: Path,
    payload: dict,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    paper_id: str,
) -> list[DimensionEvaluation]:
    if not checkpoint_path.exists():
        return []

    payload = json.loads(
        checkpoint_path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("paper_id") != paper_id:
        raise ValueError(
            "Checkpoint paper_id does not match "
            f"the current paper: {checkpoint_path}"
        )

    raw_results = payload.get(
        "dimension_results",
        [],
    )

    if not isinstance(raw_results, list):
        raise ValueError(
            "Checkpoint dimension_results must be a list."
        )

    return [
        DimensionEvaluation.model_validate(item)
        for item in raw_results
    ]


def initialize_checkpoint(
    checkpoint_path: Path,
    paper_id: str,
) -> list[DimensionEvaluation]:
    completed_results = load_checkpoint(
        checkpoint_path,
        paper_id,
    )

    if not checkpoint_path.exists():
        write_checkpoint(
            checkpoint_path,
            {
                "paper_id": paper_id,
                "completed": False,
                "dimension_results": [],
            },
        )

    return completed_results


def load_checkpoint_external_evidence(
    checkpoint_path: Path,
    paper_id: str,
) -> ExternalEvidenceSummary | None:
    if not checkpoint_path.exists():
        return None
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("paper_id") != paper_id:
        raise ValueError(
            "Checkpoint paper_id does not match "
            f"the current paper: {checkpoint_path}"
        )
    raw_summary = payload.get("external_evidence")
    if raw_summary is None:
        return None
    return ExternalEvidenceSummary.model_validate(raw_summary)


async def run_academic_evaluation(
    *,
    paper_id: str,
    title: str | None = None,
    pdf_path: str,
    source_url: str | None,
    model_name: str,
    max_chunks: int,
    checkpoint_path: str,
    hybrid_retrieval_enabled: bool = False,
    hybrid_candidate_top_k: int = 20,
    hybrid_rrf_k: int = 60,
    embedding_model: str | None = None,
    external_evidence_enabled: bool = False,
    external_evidence_limit: int = 5,
    external_evidence_retriever: (
        ExternalScholarlyEvidenceRetriever | None
    ) = None,
) -> dict:
    resolved_pdf_path = (
        Path(pdf_path)
        .expanduser()
        .resolve()
    )

    if not resolved_pdf_path.exists():
        raise FileNotFoundError(
            f"Academic PDF does not exist: "
            f"{resolved_pdf_path}"
        )

    if not resolved_pdf_path.is_file():
        raise ValueError(
            f"Academic PDF path is not a file: "
            f"{resolved_pdf_path}"
        )

    resolved_checkpoint_path = (
        Path(checkpoint_path)
        .expanduser()
        .resolve()
    )

    completed_dimension_results = (
        initialize_checkpoint(
            resolved_checkpoint_path,
            paper_id,
        )
    )

    external_evidence = load_checkpoint_external_evidence(
        resolved_checkpoint_path,
        paper_id,
    )
    if external_evidence is None:
        if external_evidence_enabled:
            try:
                evidence_retriever = (
                    external_evidence_retriever
                    or ExternalScholarlyEvidenceRetriever(
                        AcademicSearchFallback()
                    )
                )
                external_evidence = await evidence_retriever.retrieve(
                    paper_id=paper_id,
                    title=title,
                    limit=external_evidence_limit,
                )
            except Exception as exc:
                external_evidence = ExternalEvidenceSummary(
                    status="unavailable",
                    query=(title or paper_id).strip() or None,
                    warnings=[
                        "External scholarly retrieval could not be configured; "
                        "internal-evidence scoring continued.",
                        f"error_type={type(exc).__name__}",
                    ],
                )
        else:
            external_evidence = ExternalEvidenceSummary(
                status="not_requested",
                query=(title or paper_id).strip() or None,
                warnings=["External scholarly retrieval is disabled."],
            )

        write_checkpoint(
            resolved_checkpoint_path,
            {
                "paper_id": paper_id,
                "completed": False,
                "dimension_results": [
                    item.model_dump(mode="json")
                    for item in completed_dimension_results
                ],
                "external_evidence": external_evidence.model_dump(mode="json"),
            },
        )

    latest_dimension_results = list(
        completed_dimension_results
    )

    print(
        f"[evaluation] paper_id={paper_id}"
    )
    print(
        f"[evaluation] model={model_name}"
    )
    print(
        f"[evaluation] max_chunks={max_chunks}"
    )
    print(
        "[evaluation] evidence_selector="
        + ("hybrid" if hybrid_retrieval_enabled else "section")
    )
    if hybrid_retrieval_enabled:
        print(
            "[evaluation] hybrid_config="
            f"candidate_top_k={hybrid_candidate_top_k},"
            f"rrf_k={hybrid_rrf_k},embedding_model={embedding_model}"
        )
    print(
        "[evaluation] checkpoint="
        f"{resolved_checkpoint_path}"
    )
    print(
        "[evaluation] completed dimensions="
        f"{len(completed_dimension_results)}"
    )
    print(
        "[evaluation] external_evidence="
        f"{external_evidence.status},records={len(external_evidence.records)}"
    )

    # ---------------------------------------------------------
    # 1. PDF -> Evidence Chunks
    # ---------------------------------------------------------

    chunker = AcademicPaperChunker(
        target_tokens=800,
        overlap_tokens=120,
        min_chunk_tokens=100,
    )

    chunks = chunker.chunk_pdf(
        file_path=resolved_pdf_path,
        paper_id=paper_id,
        source_url=source_url,
    )

    print(
        f"[evaluation] PDF chunks={len(chunks)}"
    )

    # ---------------------------------------------------------
    # 2. Ollama evaluation client
    # ---------------------------------------------------------

    evaluation_client = (
        OllamaPaperEvaluatorClient(
            model=model_name,
        )
    )

    # ---------------------------------------------------------
    # 3. Citation validation
    # ---------------------------------------------------------

    citation_client = OllamaCitationClient()

    citation_validator = CitationValidator(
        citation_client
    )

    # ---------------------------------------------------------
    # 4. Paper evaluation pipeline
    # ---------------------------------------------------------

    embedding_client = (
        OllamaEmbeddingClient(model=embedding_model)
        if hybrid_retrieval_enabled
        else None
    )

    pipeline = PaperEvaluationPipeline(
        client=evaluation_client,
        max_chunks_per_dimension=max_chunks,
        citation_validator=citation_validator,
        max_claims_per_dimension=1,
        hybrid_retrieval_enabled=hybrid_retrieval_enabled,
        hybrid_candidate_top_k=hybrid_candidate_top_k,
        hybrid_rrf_k=hybrid_rrf_k,
        embedding_client=embedding_client,
    )

    # ---------------------------------------------------------
    # 5. Save after every completed dimension
    # ---------------------------------------------------------

    def save_dimension_checkpoint(
        results,
    ) -> None:
        nonlocal latest_dimension_results

        latest_dimension_results = list(
            results
        )

        write_checkpoint(
            resolved_checkpoint_path,
            {
                "paper_id": paper_id,
                "completed": False,
                "dimension_results": [
                    item.model_dump(
                        mode="json"
                    )
                    for item in results
                ],
                "external_evidence": external_evidence.model_dump(mode="json"),
            },
        )

        print(
            "[evaluation] checkpoint saved: "
            f"{len(results)}/9 dimensions"
        )

    # ---------------------------------------------------------
    # 6. Real nine-dimension evaluation
    # ---------------------------------------------------------

    result = await pipeline.evaluate(
        paper_id=paper_id,
        evidence_chunks=chunks,
        completed_dimension_results=(
            completed_dimension_results
        ),
        on_dimension_complete=(
            save_dimension_checkpoint
        ),
        external_evidence=external_evidence,
    )

    result_json = result.model_dump(
        mode="json"
    )

    # ---------------------------------------------------------
    # 7. Mark checkpoint complete
    # ---------------------------------------------------------

    write_checkpoint(
        resolved_checkpoint_path,
        {
            "paper_id": paper_id,
            "completed": True,
            "dimension_results": [
                item.model_dump(
                    mode="json"
                )
                for item in latest_dimension_results
            ],
            "final_evaluation": result_json,
            "external_evidence": external_evidence.model_dump(mode="json"),
        },
    )

    print(
        "[evaluation] full evaluation completed."
    )

    return result_json
