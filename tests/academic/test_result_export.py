from datetime import datetime
from types import SimpleNamespace

from backend.academic_api.services.result_export import (
    ASSESSMENT_LABEL,
    EXPORT_SCHEMA_VERSION,
    build_result_export,
    render_result_pdf,
)


def export_document() -> dict:
    job = SimpleNamespace(
        job_id="job-export",
        status="completed",
        completed_at=datetime(2026, 8, 26, 10, 0, 0),
        model_name="qwen3.5:9b",
        max_chunks=4,
        retry_count=0,
        result_json={
            "overall_score": 88,
            "checkpoint_path": "/private/checkpoint.json",
            "external_evidence": {
                "status": "available",
                "provider_used": "openalex",
                "records": [{"external_evidence_id": "external:prior"}],
            },
            "dimensions": [
                {
                    "dimension": "methodological_rigor",
                    "score": 4,
                    "rationale": "Methods are clearly described.",
                    "evidence_chunk_ids": ["paper-p2-c3"],
                    "external_evidence_ids": ["external:prior"],
                }
            ],
        },
    )
    paper = SimpleNamespace(
        paper_id="arxiv:export",
        title="Export paper",
        source_url="https://arxiv.org/abs/export",
    )
    return build_result_export(job=job, paper=paper)


def test_json_export_is_versioned_labeled_and_sanitized() -> None:
    exported = export_document()

    assert exported["schema_version"] == EXPORT_SCHEMA_VERSION
    assert exported["assessment_label"] == ASSESSMENT_LABEL
    assert "checkpoint_path" not in exported["result"]
    assert "pdf_path" not in exported["job"]


def test_pdf_export_is_a_valid_pdf_with_no_private_path() -> None:
    payload = render_result_pdf(export_document())

    assert payload.startswith(b"%PDF-")
    assert b"/private/checkpoint.json" not in payload
