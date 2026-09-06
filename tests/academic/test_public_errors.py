from backend.academic_api.public_errors import (
    internal_error_fingerprint,
    public_job_error,
    sanitize_public_payload,
)
from backend.academic_api.schemas import EvaluationJobResponse


def test_public_error_does_not_expose_path_secret_or_traceback() -> None:
    private = RuntimeError(
        "Traceback: failed at /Users/mia/private.pdf with token=super-secret"
    )

    public = public_job_error(private)

    assert public.code == "evaluation_failed"
    assert public.message == "The academic evaluation failed."
    assert "/Users/" not in public.message
    assert "super-secret" not in public.message
    assert "Traceback" not in public.message
    assert len(internal_error_fingerprint(private)) == 16


def test_known_error_categories_have_stable_public_codes() -> None:
    assert public_job_error(FileNotFoundError("/private/paper.pdf")).code == (
        "managed_pdf_unavailable"
    )
    assert public_job_error(TimeoutError("api-key=secret")).code == (
        "evaluation_dependency_unavailable"
    )
    assert public_job_error(ValueError("/internal/checkpoint.json")).code == (
        "evaluation_input_invalid"
    )

    class OllamaStructuredClientError(RuntimeError):
        pass

    model_error = public_job_error(
        OllamaStructuredClientError("invalid JSON at /private/model-output")
    )
    assert model_error.code == "model_failure"
    assert model_error.message == (
        "The evaluation model could not produce a valid response."
    )
    assert "/private/" not in model_error.message


def test_public_job_response_has_no_server_path_field() -> None:
    assert "checkpoint_path" not in EvaluationJobResponse.model_fields


def test_public_payload_removes_private_fields_and_redacts_nested_values() -> None:
    result = sanitize_public_payload(
        {
            "score": 80,
            "checkpoint_path": "/Users/mia/checkpoint.json",
            "details": {
                "text": "read /private/data/paper.pdf token=abc123",
                "traceback": "private stack",
            },
        }
    )

    assert result["score"] == 80
    assert "checkpoint_path" not in result
    assert "traceback" not in result["details"]
    assert "/private/" not in result["details"]["text"]
    assert "abc123" not in result["details"]["text"]
