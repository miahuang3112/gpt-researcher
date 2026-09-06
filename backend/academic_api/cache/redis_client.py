from __future__ import annotations

from dataclasses import dataclass
import os
import socket

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from backend.academic_api.config import settings


EVALUATION_STREAM_KEY = "academic:evaluation:stream"
EVALUATION_CONSUMER_GROUP = "academic-workers"
EVALUATION_DLQ_KEY = "academic:evaluation:dlq"
DEFAULT_CONSUMER_NAME = f"{socket.gethostname()}-{os.getpid()}"

redis_client = Redis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
)


async def check_redis_connection() -> bool:
    return bool(await redis_client.ping())


@dataclass(frozen=True)
class EvaluationMessage:
    message_id: str
    job_id: str
    event_id: str
    correlation_id: str
    delivery_count: int


async def ensure_evaluation_consumer_group() -> None:
    try:
        await redis_client.xgroup_create(
            EVALUATION_STREAM_KEY,
            EVALUATION_CONSUMER_GROUP,
            id="0-0",
            mkstream=True,
        )
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def enqueue_evaluation(
    job_id: str,
    event_id: str = "",
    correlation_id: str = "",
) -> str:
    return await redis_client.xadd(
        EVALUATION_STREAM_KEY,
        {
            "job_id": job_id,
            "event_id": event_id,
            "correlation_id": correlation_id,
        },
    )


async def _delivery_count(message_id: str) -> int:
    pending = await redis_client.xpending_range(
        EVALUATION_STREAM_KEY,
        EVALUATION_CONSUMER_GROUP,
        min=message_id,
        max=message_id,
        count=1,
    )

    if not pending:
        return 1

    return int(pending[0].get("times_delivered", 1))


def _message_from_stream(
    message_id: str,
    fields: dict[str, str],
    *,
    delivery_count: int,
) -> EvaluationMessage:
    return EvaluationMessage(
        message_id=message_id,
        job_id=fields.get("job_id", ""),
        event_id=fields.get("event_id", ""),
        correlation_id=fields.get("correlation_id", ""),
        delivery_count=delivery_count,
    )


async def dequeue_evaluation(
    *,
    consumer_name: str = DEFAULT_CONSUMER_NAME,
) -> EvaluationMessage | None:
    claimed = await redis_client.xautoclaim(
        EVALUATION_STREAM_KEY,
        EVALUATION_CONSUMER_GROUP,
        consumer_name,
        min_idle_time=settings.stream_claim_idle_milliseconds,
        start_id="0-0",
        count=1,
    )

    claimed_messages = claimed[1] if len(claimed) > 1 else []
    if claimed_messages:
        message_id, fields = claimed_messages[0]
        return _message_from_stream(
            message_id,
            fields,
            delivery_count=await _delivery_count(message_id),
        )

    response = await redis_client.xreadgroup(
        EVALUATION_CONSUMER_GROUP,
        consumer_name,
        streams={EVALUATION_STREAM_KEY: ">"},
        count=1,
        block=settings.stream_block_milliseconds,
    )

    if not response:
        return None

    _, messages = response[0]
    if not messages:
        return None

    message_id, fields = messages[0]
    return _message_from_stream(
        message_id,
        fields,
        delivery_count=1,
    )


async def acknowledge_evaluation(message_id: str) -> int:
    return int(
        await redis_client.xack(
            EVALUATION_STREAM_KEY,
            EVALUATION_CONSUMER_GROUP,
            message_id,
        )
    )


async def dead_letter_evaluation(
    message: EvaluationMessage,
    error_message: str,
) -> None:
    pipeline = redis_client.pipeline(transaction=True)
    pipeline.xadd(
        EVALUATION_DLQ_KEY,
        {
            "source_message_id": message.message_id,
            "job_id": message.job_id,
            "event_id": message.event_id,
            "correlation_id": message.correlation_id,
            "delivery_count": str(message.delivery_count),
            "error": error_message[:1000],
        },
    )
    pipeline.xack(
        EVALUATION_STREAM_KEY,
        EVALUATION_CONSUMER_GROUP,
        message.message_id,
    )
    await pipeline.execute()



async def close_redis_connection() -> None:
    await redis_client.aclose()
