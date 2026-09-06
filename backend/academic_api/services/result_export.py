from __future__ import annotations

import textwrap
from datetime import datetime, timezone

import fitz

from backend.academic_api.public_errors import sanitize_public_payload


EXPORT_SCHEMA_VERSION = "academic-evaluation-export.v1"
ASSESSMENT_LABEL = "Automated academic assessment — not peer review."


def build_result_export(*, job, paper) -> dict:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assessment_label": ASSESSMENT_LABEL,
        "job": {
            "job_id": job.job_id,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "source_url": paper.source_url,
            "status": job.status,
            "completed_at": (
                job.completed_at.isoformat() if job.completed_at else None
            ),
            "model_name": job.model_name,
            "max_chunks": job.max_chunks,
            "retry_count": job.retry_count or 0,
        },
        "result": sanitize_public_payload(job.result_json),
    }


def render_result_pdf(export: dict) -> bytes:
    result = export.get("result") or {}
    job = export["job"]
    document = fitz.open()
    page = document.new_page()
    y = 54.0

    def write(text: object, *, size: float = 10, gap: float = 5) -> None:
        nonlocal page, y
        normalized = str(text or "").replace("\x00", " ")
        for raw_line in normalized.splitlines() or [""]:
            for line in textwrap.wrap(raw_line, width=92) or [""]:
                if y > 790:
                    page = document.new_page()
                    y = 54.0
                page.insert_text((54, y), line, fontsize=size, fontname="helv")
                y += size + gap

    write("Academic Paper Evaluation", size=18, gap=10)
    write(ASSESSMENT_LABEL, size=9, gap=9)
    write(f"Paper: {job.get('title') or job.get('paper_id')}", size=12)
    write(f"Job ID: {job.get('job_id')}")
    write(f"Schema: {export.get('schema_version')}")
    write(f"Overall score: {result.get('overall_score', 'N/A')}", size=14)
    write(f"Evidence coverage: {result.get('evidence_coverage', 'N/A')}")
    external = result.get("external_evidence") or {}
    write(
        "External scholarly evidence: "
        f"{external.get('status', 'not recorded')} "
        f"(provider={external.get('provider_used') or 'none'}, "
        f"records={len(external.get('records') or [])})"
    )
    y += 8

    for dimension in result.get("dimensions", []):
        write(
            f"{dimension.get('dimension', 'dimension')} — "
            f"{dimension.get('score', 'N/A')}/5",
            size=12,
        )
        write(dimension.get("rationale") or "No rationale provided.")
        evidence = dimension.get("evidence_chunk_ids") or []
        if evidence:
            write("Evidence: " + ", ".join(map(str, evidence)), size=8)
        external_ids = dimension.get("external_evidence_ids") or []
        if external_ids:
            write(
                "External scholarly evidence: "
                + ", ".join(map(str, external_ids)),
                size=8,
            )
        y += 6

    payload = document.tobytes(garbage=4, deflate=True)
    document.close()
    return payload
