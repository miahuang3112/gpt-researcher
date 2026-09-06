from __future__ import annotations

import json

import pytest

from backend.academic_api import health
from backend.academic_api import main as api_main


@pytest.mark.asyncio
async def test_api_readiness_reports_mysql_and_redis(monkeypatch) -> None:
    async def healthy() -> bool:
        return True

    monkeypatch.setattr(health, "check_mysql", healthy)
    monkeypatch.setattr(health, "check_redis", healthy)

    report = await health.api_readiness_report()

    assert report == {
        "status": "ready",
        "checks": {
            "mysql": {"status": "ok"},
            "redis": {"status": "ok"},
        },
    }


@pytest.mark.asyncio
async def test_worker_readiness_reports_ollama_failure(monkeypatch) -> None:
    async def healthy() -> bool:
        return True

    async def unavailable_ollama() -> bool:
        raise ConnectionError("ollama unavailable")

    monkeypatch.setattr(health, "check_mysql", healthy)
    monkeypatch.setattr(health, "check_redis", healthy)
    monkeypatch.setattr(health, "check_ollama", unavailable_ollama)

    report = await health.worker_readiness_report()

    assert report["status"] == "not_ready"
    assert report["checks"]["mysql"] == {"status": "ok"}
    assert report["checks"]["redis"] == {"status": "ok"}
    assert report["checks"]["ollama"] == {
        "status": "error",
        "error_type": "ConnectionError",
    }


@pytest.mark.asyncio
async def test_liveness_stays_alive_when_readiness_fails(monkeypatch) -> None:
    async def not_ready() -> dict:
        return {
            "status": "not_ready",
            "checks": {
                "mysql": {"status": "ok"},
                "redis": {
                    "status": "error",
                    "error_type": "ConnectionError",
                },
            },
        }

    monkeypatch.setattr(api_main, "api_readiness_report", not_ready)

    assert await api_main.liveness() == {"status": "alive"}
    response = await api_main.readiness()
    assert response.status_code == 503
    assert json.loads(response.body)["status"] == "not_ready"


@pytest.mark.asyncio
async def test_legacy_health_endpoint_remains_compatible() -> None:
    assert await api_main.health() == {"status": "ok"}
