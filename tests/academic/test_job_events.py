from __future__ import annotations

import json

import pytest

from backend.academic_api.services.job_events import (
    encode_sse,
    evaluation_event_stream,
    parse_last_event_id,
)


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


def test_last_event_id_and_encoding_are_stable() -> None:
    assert parse_last_event_id("12") == 12
    assert parse_last_event_id("invalid") == 0
    encoded = encode_sse(
        event_id=3,
        event="progress",
        data={"status": "running"},
    )
    assert encoded.startswith("id: 3\nevent: progress\ndata: ")
    assert json.loads(encoded.split("data: ", 1)[1]) == {
        "status": "running"
    }


@pytest.mark.asyncio
async def test_reconnect_replays_current_terminal_snapshot() -> None:
    async def load(job_id: str):
        return {
            "job_id": job_id,
            "status": "completed",
            "state_version": 7,
        }

    events = [
        event
        async for event in evaluation_event_stream(
            request=ConnectedRequest(),
            job_id="job-7",
            last_event_id="7",
            snapshot_loader=load,
            poll_seconds=0,
        )
    ]

    assert len(events) == 1
    assert events[0].startswith("id: 7\n")
    assert '"status":"completed"' in events[0]


@pytest.mark.asyncio
async def test_stream_emits_only_new_versions_until_terminal() -> None:
    snapshots = [
        {"job_id": "job", "status": "running", "state_version": 2},
        {"job_id": "job", "status": "running", "state_version": 2},
        {"job_id": "job", "status": "completed", "state_version": 3},
    ]

    async def load(job_id: str):
        return snapshots.pop(0)

    async def no_wait(seconds: float):
        return None

    events = [
        event
        async for event in evaluation_event_stream(
            request=ConnectedRequest(),
            job_id="job",
            last_event_id="1",
            snapshot_loader=load,
            poll_seconds=0,
            sleep=no_wait,
        )
    ]

    assert [event.splitlines()[0] for event in events] == ["id: 2", "id: 3"]
