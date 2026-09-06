from __future__ import annotations

import json
import logging

import pytest
from starlette.requests import Request
from starlette.responses import Response

from backend.academic_api import main as api_main
from backend.academic_api.observability import (
    AcademicJsonFormatter,
    correlation_context,
    normalize_correlation_id,
)


def test_json_formatter_includes_correlation_and_structured_fields() -> None:
    record = logging.LogRecord(
        name="academic.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="ignored message",
        args=(),
        exc_info=None,
    )
    record.event_name = "worker.stage.completed"
    record.structured_fields = {
        "job_id": "job-1",
        "stage": "evaluate",
        "duration_ms": 12.5,
    }

    with correlation_context("request-1"):
        payload = json.loads(AcademicJsonFormatter().format(record))

    assert payload["event"] == "worker.stage.completed"
    assert payload["correlation_id"] == "request-1"
    assert payload["job_id"] == "job-1"
    assert payload["stage"] == "evaluate"
    assert payload["duration_ms"] == 12.5


def test_correlation_id_is_sanitized_and_bounded() -> None:
    value = normalize_correlation_id(" request/<unsafe> " + "x" * 200)

    assert "/" not in value
    assert "<" not in value
    assert len(value) == 128
    assert len(normalize_correlation_id(None)) == 32


@pytest.mark.asyncio
async def test_request_middleware_echoes_correlation_id() -> None:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/health/live",
            "headers": [(b"x-request-id", b"resume-request-1")],
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 123),
            "scheme": "http",
        }
    )

    async def call_next(received_request: Request) -> Response:
        assert received_request is request
        return Response(status_code=200)

    response = await api_main.request_observability(request, call_next)

    assert response.headers["X-Request-ID"] == "resume-request-1"
