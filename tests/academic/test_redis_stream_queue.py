from __future__ import annotations

import pytest
from redis.exceptions import ResponseError

from backend.academic_api.cache import redis_client as queue


class FakePipeline:
    def __init__(self) -> None:
        self.commands = []
        self.executed = False

    def xadd(self, stream, fields):
        self.commands.append(("xadd", stream, fields))
        return self

    def xack(self, stream, group, message_id):
        self.commands.append(("xack", stream, group, message_id))
        return self

    async def execute(self):
        self.executed = True
        return ["2-0", 1]


class FakeRedis:
    def __init__(self) -> None:
        self.group_error = None
        self.claimed = ["0-0", [], []]
        self.new_messages = []
        self.pending = []
        self.xadd_calls = []
        self.xack_calls = []
        self.pipeline_instance = FakePipeline()

    async def xgroup_create(self, *args, **kwargs):
        if self.group_error is not None:
            raise self.group_error
        return True

    async def xadd(self, stream, fields):
        self.xadd_calls.append((stream, fields))
        return "1-0"

    async def xautoclaim(self, *args, **kwargs):
        return self.claimed

    async def xreadgroup(self, *args, **kwargs):
        return self.new_messages

    async def xpending_range(self, *args, **kwargs):
        return self.pending

    async def xack(self, stream, group, message_id):
        self.xack_calls.append((stream, group, message_id))
        return 1

    def pipeline(self, transaction=True):
        assert transaction is True
        return self.pipeline_instance


@pytest.mark.asyncio
async def test_consumer_group_creation_is_idempotent(monkeypatch) -> None:
    fake = FakeRedis()
    fake.group_error = ResponseError("BUSYGROUP Consumer Group name already exists")
    monkeypatch.setattr(queue, "redis_client", fake)

    await queue.ensure_evaluation_consumer_group()


@pytest.mark.asyncio
async def test_enqueue_and_read_new_stream_message(monkeypatch) -> None:
    fake = FakeRedis()
    fake.new_messages = [
        [
            queue.EVALUATION_STREAM_KEY,
            [
                (
                    "10-0",
                    {
                        "job_id": "job-10",
                        "event_id": "event-10",
                        "correlation_id": "request-10",
                    },
                )
            ],
        ]
    ]
    monkeypatch.setattr(queue, "redis_client", fake)

    message_id = await queue.enqueue_evaluation(
        "job-10",
        "event-10",
        "request-10",
    )
    message = await queue.dequeue_evaluation(consumer_name="worker-a")

    assert message_id == "1-0"
    assert fake.xadd_calls == [
        (
            queue.EVALUATION_STREAM_KEY,
            {
                "job_id": "job-10",
                "event_id": "event-10",
                "correlation_id": "request-10",
            },
        )
    ]
    assert message is not None
    assert message.message_id == "10-0"
    assert message.job_id == "job-10"
    assert message.event_id == "event-10"
    assert message.correlation_id == "request-10"
    assert message.delivery_count == 1


@pytest.mark.asyncio
async def test_stale_pending_message_reports_delivery_count(monkeypatch) -> None:
    fake = FakeRedis()
    fake.claimed = [
        "0-0",
        [
            (
                "11-0",
                {
                    "job_id": "job-11",
                    "event_id": "event-11",
                    "correlation_id": "request-11",
                },
            )
        ],
        [],
    ]
    fake.pending = [{"message_id": "11-0", "times_delivered": 3}]
    monkeypatch.setattr(queue, "redis_client", fake)

    message = await queue.dequeue_evaluation(consumer_name="worker-b")

    assert message is not None
    assert message.message_id == "11-0"
    assert message.delivery_count == 3


@pytest.mark.asyncio
async def test_dead_letter_add_and_ack_are_atomic(monkeypatch) -> None:
    fake = FakeRedis()
    monkeypatch.setattr(queue, "redis_client", fake)
    message = queue.EvaluationMessage(
        message_id="12-0",
        job_id="job-12",
        event_id="event-12",
        correlation_id="request-12",
        delivery_count=3,
    )

    await queue.dead_letter_evaluation(message, "permanent failure")

    assert fake.pipeline_instance.executed is True
    assert fake.pipeline_instance.commands[0][0:2] == (
        "xadd",
        queue.EVALUATION_DLQ_KEY,
    )
    assert fake.pipeline_instance.commands[1] == (
        "xack",
        queue.EVALUATION_STREAM_KEY,
        queue.EVALUATION_CONSUMER_GROUP,
        "12-0",
    )
