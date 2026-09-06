from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from redis.asyncio import Redis
from redis.exceptions import ResponseError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.academic_api.cache.redis_client import (
    EVALUATION_CONSUMER_GROUP,
    EVALUATION_DLQ_KEY,
    EVALUATION_STREAM_KEY,
)
from backend.academic_api.models.evaluation_job import EvaluationJob


PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
JOB_STATUSES = (
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "completed",
    "failed",
)
MODEL_ERROR_CODES = {
    "model_failure",
}


@dataclass(frozen=True)
class AcademicMetricsSnapshot:
    jobs_by_status: dict[str, int]
    duration_seconds: tuple[float, ...] = ()
    retry_events: int = 0
    model_failures: int = 0
    queue_stream_length: int = 0
    queue_pending: int = 0
    queue_dlq_length: int = 0
    mysql_up: bool = True
    redis_up: bool = True


def _metric(name: str, value: int | float, labels: str = "") -> str:
    return f"{name}{labels} {value}"


def render_prometheus(snapshot: AcademicMetricsSnapshot) -> str:
    durations = snapshot.duration_seconds
    duration_sum = sum(durations)
    duration_max = max(durations, default=0.0)
    duration_avg = duration_sum / len(durations) if durations else 0.0

    lines = [
        "# HELP academic_jobs Current evaluation jobs by persisted status.",
        "# TYPE academic_jobs gauge",
    ]
    lines.extend(
        _metric(
            "academic_jobs",
            snapshot.jobs_by_status.get(status, 0),
            f'{{status="{status}"}}',
        )
        for status in JOB_STATUSES
    )
    lines.extend(
        [
            "# HELP academic_queue_stream_length Redis stream entries.",
            "# TYPE academic_queue_stream_length gauge",
            _metric("academic_queue_stream_length", snapshot.queue_stream_length),
            "# HELP academic_queue_pending Unacknowledged consumer-group entries.",
            "# TYPE academic_queue_pending gauge",
            _metric("academic_queue_pending", snapshot.queue_pending),
            "# HELP academic_queue_dlq_length Dead-letter stream entries.",
            "# TYPE academic_queue_dlq_length gauge",
            _metric("academic_queue_dlq_length", snapshot.queue_dlq_length),
            "# HELP academic_retry_events_total Persisted retry attempts across jobs.",
            "# TYPE academic_retry_events_total gauge",
            _metric("academic_retry_events_total", snapshot.retry_events),
            "# HELP academic_model_failures_total Failed jobs with a model/provider error code.",
            "# TYPE academic_model_failures_total gauge",
            _metric("academic_model_failures_total", snapshot.model_failures),
            "# HELP academic_evaluation_duration_seconds Completed job runtime summary.",
            "# TYPE academic_evaluation_duration_seconds summary",
            _metric("academic_evaluation_duration_seconds_count", len(durations)),
            _metric("academic_evaluation_duration_seconds_sum", duration_sum),
            "# HELP academic_evaluation_duration_seconds_avg Average completed job runtime.",
            "# TYPE academic_evaluation_duration_seconds_avg gauge",
            _metric("academic_evaluation_duration_seconds_avg", duration_avg),
            "# HELP academic_evaluation_duration_seconds_max Maximum completed job runtime.",
            "# TYPE academic_evaluation_duration_seconds_max gauge",
            _metric("academic_evaluation_duration_seconds_max", duration_max),
            "# HELP academic_metrics_mysql_up Whether the MySQL metrics query succeeded.",
            "# TYPE academic_metrics_mysql_up gauge",
            _metric("academic_metrics_mysql_up", int(snapshot.mysql_up)),
            "# HELP academic_metrics_redis_up Whether the Redis metrics query succeeded.",
            "# TYPE academic_metrics_redis_up gauge",
            _metric("academic_metrics_redis_up", int(snapshot.redis_up)),
        ]
    )
    return "\n".join(lines) + "\n"


def _durations(rows: Iterable[tuple[datetime | None, datetime | None]]) -> tuple[float, ...]:
    return tuple(
        max(0.0, (completed_at - started_at).total_seconds())
        for started_at, completed_at in rows
        if started_at is not None and completed_at is not None
    )


async def collect_academic_metrics(
    session: AsyncSession,
    redis: Redis,
) -> AcademicMetricsSnapshot:
    jobs_by_status = {status: 0 for status in JOB_STATUSES}
    durations: tuple[float, ...] = ()
    retry_events = 0
    model_failures = 0
    mysql_up = True
    redis_up = True

    try:
        status_rows = await session.execute(
            select(EvaluationJob.status, func.count(EvaluationJob.id)).group_by(
                EvaluationJob.status
            )
        )
        for status, count in status_rows.all():
            jobs_by_status[str(status)] = int(count)

        duration_rows = await session.execute(
            select(EvaluationJob.started_at, EvaluationJob.completed_at).where(
                EvaluationJob.status == "completed"
            )
        )
        durations = _durations(duration_rows.all())
        retry_events = int(
            await session.scalar(select(func.coalesce(func.sum(EvaluationJob.retry_count), 0)))
            or 0
        )
        model_failures = int(
            await session.scalar(
                select(func.count(EvaluationJob.id)).where(
                    EvaluationJob.status == "failed",
                    EvaluationJob.error_code.in_(MODEL_ERROR_CODES),
                )
            )
            or 0
        )
    except Exception:
        mysql_up = False

    stream_length = pending = dlq_length = 0
    try:
        stream_length = int(await redis.xlen(EVALUATION_STREAM_KEY))
        dlq_length = int(await redis.xlen(EVALUATION_DLQ_KEY))
        try:
            pending_summary = await redis.xpending(
                EVALUATION_STREAM_KEY,
                EVALUATION_CONSUMER_GROUP,
            )
            pending = int(pending_summary.get("pending", 0))
        except ResponseError as exc:
            if "NOGROUP" not in str(exc):
                raise
    except Exception:
        redis_up = False

    return AcademicMetricsSnapshot(
        jobs_by_status=jobs_by_status,
        duration_seconds=durations,
        retry_events=retry_events,
        model_failures=model_failures,
        queue_stream_length=stream_length,
        queue_pending=pending,
        queue_dlq_length=dlq_length,
        mysql_up=mysql_up,
        redis_up=redis_up,
    )
