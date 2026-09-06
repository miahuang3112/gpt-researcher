from __future__ import annotations

from datetime import datetime

import pytest

from backend.academic_api.services.outbox import (
    EVALUATION_REQUESTED_EVENT,
    build_evaluation_requested_event,
    dispatch_next_outbox_event,
    retry_delay_seconds,
)


class FakeResult:
    def __init__(self, event) -> None:
        self.event = event

    def scalar_one_or_none(self):
        return self.event


class FakeSession:
    def __init__(self, event) -> None:
        self.event = event
        self.commit_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def execute(self, statement) -> FakeResult:
        return FakeResult(self.event)

    async def commit(self) -> None:
        self.commit_count += 1


def test_build_evaluation_requested_event() -> None:
    event = build_evaluation_requested_event(
        job_database_id=42,
        public_job_id="public-job-42",
    )

    assert event.job_id == 42
    assert event.event_type == EVALUATION_REQUESTED_EVENT
    assert event.payload_json == {
        "job_id": "public-job-42",
        "correlation_id": "public-job-42",
    }
    assert event.attempt_count == 0
    assert event.published_at is None


@pytest.mark.asyncio
async def test_dispatch_marks_event_published_after_enqueue() -> None:
    event = build_evaluation_requested_event(
        job_database_id=1,
        public_job_id="job-success",
    )
    session = FakeSession(event)
    enqueued = []
    now = datetime(2026, 8, 26, 12, 0, 0)

    async def fake_enqueue(
        job_id: str,
        event_id: str,
        correlation_id: str,
    ) -> None:
        enqueued.append((job_id, event_id, correlation_id))

    result = await dispatch_next_outbox_event(
        session_factory=lambda: session,
        enqueue=fake_enqueue,
        now_factory=lambda: now,
    )

    assert result.status == "published"
    assert enqueued == [
        ("job-success", event.event_id, "job-success")
    ]
    assert event.published_at == now
    assert event.last_error is None
    assert session.commit_count == 1


@pytest.mark.asyncio
async def test_dispatch_persists_retry_when_enqueue_fails() -> None:
    event = build_evaluation_requested_event(
        job_database_id=2,
        public_job_id="job-retry",
    )
    session = FakeSession(event)
    now = datetime(2026, 8, 26, 12, 0, 0)

    async def failing_enqueue(
        job_id: str,
        event_id: str,
        correlation_id: str,
    ) -> None:
        raise ConnectionError("redis unavailable")

    result = await dispatch_next_outbox_event(
        session_factory=lambda: session,
        enqueue=failing_enqueue,
        now_factory=lambda: now,
    )

    assert result.status == "failed"
    assert event.published_at is None
    assert event.attempt_count == 1
    assert event.next_attempt_at > now
    assert event.last_error == "ConnectionError: redis unavailable"
    assert session.commit_count == 1


def test_retry_delay_is_exponential_and_bounded() -> None:
    assert [retry_delay_seconds(attempt) for attempt in range(1, 5)] == [
        1,
        2,
        4,
        8,
    ]
    assert retry_delay_seconds(100) == 60
