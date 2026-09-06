from __future__ import annotations

from hmac import compare_digest

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.academic_api.config import settings


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Required when ACADEMIC_AUTH_MODE=api_key.",
)


def verify_academic_access(
    provided_key: str | None,
    *,
    auth_mode: str,
    configured_key: str | None,
) -> bool:
    """Evaluate the configured boundary without leaking key values."""
    if auth_mode == "trusted_local":
        return True

    if auth_mode != "api_key" or not provided_key or not configured_key:
        return False

    return compare_digest(
        provided_key.encode("utf-8"),
        configured_key.encode("utf-8"),
    )


async def require_academic_access(
    provided_key: str | None = Security(api_key_header),
) -> None:
    configured_key = (
        settings.academic_api_key.get_secret_value()
        if settings.academic_api_key is not None
        else None
    )

    if verify_academic_access(
        provided_key,
        auth_mode=settings.academic_auth_mode,
        configured_key=configured_key,
    ):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Academic API authentication failed.",
        headers={"WWW-Authenticate": "ApiKey"},
    )
