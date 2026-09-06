from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from time import perf_counter

from sqlalchemy import select

from backend.academic_api.cache.redis_client import enqueue_evaluation
from backend.academic_api.config import settings
from backend.academic_api.db.session import AsyncSessionLocal
from backend.academic_api.models.outbox_event import OutboxEvent
from backend.academic_api.observability import (
    correlation_context,
    elapsed_milliseconds,
    get_logger,
    log_event,
)


EVALUATION_REQUESTED_EVENT = "evaluation.requested"
logger = get_logger("outbox")


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def build_evaluation_requested_event(
    *,
    job_database_id: int,
    public_job_id: str,
    correlation_id: str | None = None,
) -> OutboxEvent:
    return OutboxEvent(
        event_id=uuid4().hex,
        job_id=job_database_id,
        event_type=EVALUATION_REQUESTED_EVENT,
        payload_json={
            "job_id": public_job_id,
            "correlation_id": correlation_id or public_job_id,
        },
        attempt_count=0,
        next_attempt_at=utc_now_naive(),
    )


def retry_delay_seconds(attempt_count: int) -> int:
    exponent = max(0, min(attempt_count - 1, 30))
    return min(
        2**exponent,
        settings.outbox_max_retry_seconds,
    )


@dataclass(frozen=True)
class DispatchResult:
    status: str
    event_id: str | None = None


async def dispatch_next_outbox_event(
    *,
    session_factory=AsyncSessionLocal,
    enqueue: Callable[[str, str, str], Awaitable[object]] = enqueue_evaluation,
    now_factory: Callable[[], datetime] = utc_now_naive,
) -> DispatchResult:
    now = now_factory()

    async with session_factory() as session:
        result = await session.execute(
            select(OutboxEvent)
            .where(
                OutboxEvent.published_at.is_(None),
                OutboxEvent.next_attempt_at <= now,
            )
            .order_by(OutboxEvent.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        event = result.scalar_one_or_none()

        if event is None:
            return DispatchResult(status="empty")

        event_id = event.event_id
        started_at = perf_counter()
        raw_job_id = event.payload_json.get("job_id")
        public_job_id = raw_job_id if isinstance(raw_job_id, str) else ""
        raw_correlation_id = event.payload_json.get("correlation_id")
        correlation_id = (
            raw_correlation_id
            if isinstance(raw_correlation_id, str) and raw_correlation_id
            else public_job_id or event_id
        )

        try:
            if event.event_type != EVALUATION_REQUESTED_EVENT:
                raise ValueError(
                    f"Unsupported outbox event type: {event.event_type}"
                )

            if not public_job_id:
                raise ValueError("Outbox event payload has no valid job_id.")

            with correlation_context(correlation_id):
                await enqueue(public_job_id, event.event_id, correlation_id)

        except Exception as exc:
            event.attempt_count += 1
            delay_seconds = retry_delay_seconds(event.attempt_count)
            event.next_attempt_at = now + timedelta(seconds=delay_seconds)
            event.last_error = (
                f"{type(exc).__name__}: {exc}"
            )[: settings.outbox_error_max_length]
            await session.commit()

            with correlation_context(correlation_id):
                log_event(
                    logger,
                    "outbox.publish.failed",
                    level=logging.WARNING,
                    event_id=event_id,
                    job_id=public_job_id,
                    attempt_count=event.attempt_count,
                    retry_delay_seconds=delay_seconds,
                    duration_ms=elapsed_milliseconds(started_at),
                    error_type=type(exc).__name__,
                )
            return DispatchResult(status="failed", event_id=event_id)

        event.published_at = now
        event.last_error = None
        await session.commit()

        with correlation_context(correlation_id):
            log_event(
                logger,
                "outbox.publish.completed",
                event_id=event_id,
                job_id=public_job_id,
                duration_ms=elapsed_milliseconds(started_at),
            )
        return DispatchResult(status="published", event_id=event_id)


async def dispatch_pending_outbox_events(
    *,
    limit: int | None = None,
) -> int:
    batch_size = limit or settings.outbox_batch_size
    processed = 0

    for _ in range(batch_size):
        result = await dispatch_next_outbox_event()
        if result.status == "empty":
            break
        processed += 1

    return processed


async def run_outbox_dispatcher() -> None:
    log_event(logger, "outbox.dispatcher.started")

    while True:
        try:
            processed = await dispatch_pending_outbox_events()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            processed = 0
            log_event(
                logger,
                "outbox.dispatcher.iteration_failed",
                level=logging.ERROR,
                error_type=type(exc).__name__,
            )

        if processed >= settings.outbox_batch_size:
            continue

        await asyncio.sleep(settings.outbox_poll_interval_seconds)
