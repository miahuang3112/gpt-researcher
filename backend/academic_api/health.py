from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
from sqlalchemy import text

from backend.academic_api.cache.redis_client import check_redis_connection
from backend.academic_api.config import settings
from backend.academic_api.db.session import engine


Check = Callable[[], Awaitable[bool]]


async def check_mysql() -> bool:
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT 1"))
        return result.scalar_one() == 1


async def check_redis() -> bool:
    return await check_redis_connection()


async def check_ollama() -> bool:
    async with httpx.AsyncClient(
        timeout=settings.healthcheck_timeout_seconds,
    ) as client:
        response = await client.get(
            f"{settings.ollama_base_url.rstrip('/')}/api/tags"
        )
        response.raise_for_status()
        return True


async def run_check(check: Check) -> dict[str, str]:
    try:
        healthy = await asyncio.wait_for(
            check(),
            timeout=settings.healthcheck_timeout_seconds,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
        }

    return {"status": "ok" if healthy else "error"}


async def readiness_report(*, include_ollama: bool) -> dict:
    checks: dict[str, Check] = {
        "mysql": check_mysql,
        "redis": check_redis,
    }
    if include_ollama:
        checks["ollama"] = check_ollama

    names = list(checks)
    results = await asyncio.gather(
        *(run_check(checks[name]) for name in names)
    )
    check_results = dict(zip(names, results, strict=True))
    ready = all(
        result["status"] == "ok"
        for result in check_results.values()
    )

    return {
        "status": "ready" if ready else "not_ready",
        "checks": check_results,
    }


async def api_readiness_report() -> dict:
    return await readiness_report(include_ollama=False)


async def worker_readiness_report() -> dict:
    return await readiness_report(include_ollama=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        choices=("api-ready", "worker-ready"),
    )
    args = parser.parse_args()

    report = asyncio.run(
        worker_readiness_report()
        if args.target == "worker-ready"
        else api_readiness_report()
    )
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "ready" else 1)


if __name__ == "__main__":
    main()
