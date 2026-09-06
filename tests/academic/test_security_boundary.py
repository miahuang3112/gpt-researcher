import pytest
from pydantic import ValidationError

from backend.academic_api.config import Settings
from backend.academic_api.security import verify_academic_access


def test_trusted_local_mode_is_explicitly_permissive() -> None:
    assert verify_academic_access(
        None,
        auth_mode="trusted_local",
        configured_key=None,
    )


def test_api_key_mode_requires_constant_value_match() -> None:
    assert verify_academic_access(
        "correct-secret",
        auth_mode="api_key",
        configured_key="correct-secret",
    )
    assert not verify_academic_access(
        None,
        auth_mode="api_key",
        configured_key="correct-secret",
    )
    assert not verify_academic_access(
        "wrong-secret",
        auth_mode="api_key",
        configured_key="correct-secret",
    )


def test_api_key_mode_cannot_start_without_configured_secret() -> None:
    with pytest.raises(ValidationError, match="ACADEMIC_API_KEY is required"):
        Settings(
            academic_auth_mode="api_key",
            academic_api_key=None,
            _env_file=None,
        )


def test_configured_key_is_redacted_from_settings_representation() -> None:
    configured = Settings(
        academic_auth_mode="api_key",
        academic_api_key="do-not-print-this",
        _env_file=None,
    )

    assert "do-not-print-this" not in repr(configured)
