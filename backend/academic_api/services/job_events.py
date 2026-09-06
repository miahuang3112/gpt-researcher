from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from time import monotonic


TERMINAL_EVENT_STATES = {"completed", "failed", "cancelled"}


def parse_last_event_id(value: str | None) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def encode_sse(*, event_id: int, event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event}\ndata: {payload}\n\n"


async def evaluation_event_stream(
    *,
    request,
    job_id: str,
    last_event_id: str | None,
    snapshot_loader: Callable[[str], Awaitable[dict | None]],
    poll_seconds: float = 1.0,
    heartbeat_seconds: float = 15.0,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> AsyncIterator[str]:
    emitted_version = parse_last_event_id(last_event_id)
    first_snapshot = True
    last_heartbeat = monotonic()

    while True:
        if await request.is_disconnected():
            return

        snapshot = await snapshot_loader(job_id)
        if snapshot is None:
            yield encode_sse(
                event_id=emitted_version,
                event="not_found",
                data={"job_id": job_id, "status": "not_found"},
            )
            return

        version = int(snapshot["state_version"])
        if first_snapshot or version > emitted_version:
            yield encode_sse(
                event_id=version,
                event="progress",
                data=snapshot,
            )
            first_snapshot = False
            emitted_version = max(emitted_version, version)
            if snapshot["status"] in TERMINAL_EVENT_STATES:
                return

        now = monotonic()
        if now - last_heartbeat >= heartbeat_seconds:
            yield ": keep-alive\n\n"
            last_heartbeat = now

        await sleep(poll_seconds)
