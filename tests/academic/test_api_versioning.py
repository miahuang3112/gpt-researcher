from backend.academic_api.main import app
from backend.academic_api.schemas import EvaluationJobResponse


def test_public_schema_has_explicit_stable_version() -> None:
    fields = EvaluationJobResponse.model_fields
    assert fields["schema_version"].default == "academic-evaluation.v1"


def test_openapi_publishes_canonical_v1_routes() -> None:
    paths = app.openapi()["paths"]

    assert "/api/v1/academic/evaluations" in paths
    assert "/api/v1/academic/evaluations/{job_id}/result" in paths
    assert "/api/v1/academic/evaluations/{job_id}/events" in paths
    assert "/api/v1/academic/evaluations/{job_id}/export.json" in paths
    assert "/api/v1/academic/evaluations/{job_id}/export.pdf" in paths
    assert "/api/academic/evaluations" not in paths
