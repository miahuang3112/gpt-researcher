from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PublicJobError:
    code: str
    message: str


PRIVATE_KEYS = {
    "checkpoint_path",
    "pdf_path",
    "server_path",
    "api_key",
    "password",
    "secret",
    "token",
    "traceback",
}
POSIX_PATH_PATTERN = re.compile(r"(?<![\w:])/(?:[^\s/]+/)*[^\s,;]+")
WINDOWS_PATH_PATTERN = re.compile(r"\b[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s,;]+")
SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+"
)


def public_job_error(error: Exception) -> PublicJobError:
    """Map internal exceptions to stable messages with no paths or secrets."""
    if isinstance(error, FileNotFoundError):
        return PublicJobError(
            code="managed_pdf_unavailable",
            message="The managed PDF is unavailable.",
        )

    error_type = type(error).__name__.casefold()
    if error_type.startswith("ollama"):
        return PublicJobError(
            code="model_failure",
            message="The evaluation model could not produce a valid response.",
        )
    if "timeout" in error_type or "connect" in error_type:
        return PublicJobError(
            code="evaluation_dependency_unavailable",
            message="An evaluation dependency is temporarily unavailable.",
        )

    if isinstance(error, (TypeError, ValueError)):
        return PublicJobError(
            code="evaluation_input_invalid",
            message="The evaluation input could not be processed.",
        )

    return PublicJobError(
        code="evaluation_failed",
        message="The academic evaluation failed.",
    )


def internal_error_fingerprint(error: Exception) -> str:
    """Return a correlation-safe fingerprint without exposing error content."""
    value = f"{type(error).__name__}\0{error}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def sanitize_public_payload(value):
    """Recursively remove internal fields and redact sensitive string forms."""
    if isinstance(value, dict):
        return {
            key: sanitize_public_payload(item)
            for key, item in value.items()
            if key.casefold() not in PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [sanitize_public_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_payload(item) for item in value]
    if not isinstance(value, str):
        return value

    if "traceback (most recent call last)" in value.casefold():
        return "[internal error details redacted]"

    redacted = SECRET_PATTERN.sub(r"\1=[redacted]", value)
    redacted = WINDOWS_PATH_PATTERN.sub("[server path redacted]", redacted)
    redacted = POSIX_PATH_PATTERN.sub("[server path redacted]", redacted)
    return redacted
