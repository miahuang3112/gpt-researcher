from datetime import datetime
from types import SimpleNamespace

import pytest

from backend.academic_api.job_state import (
    EvaluationState,
    InvalidJobTransition,
    transition_job,
)


def job(status: str = "queued") -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        state_version=1,
        started_at=None,
        completed_at=None,
        cancel_requested_at=None,
        cancelled_at=None,
        current_dimension="research_question_clarity",
    )


def test_valid_lifecycle_is_persisted_with_monotonic_versions() -> None:
    value = job()
    started = datetime(2026, 8, 26, 10, 0, 0)
    completed = datetime(2026, 8, 26, 10, 5, 0)

    assert transition_job(value, EvaluationState.RUNNING, now=started)
    assert value.status == "running"
    assert value.started_at == started
    assert value.state_version == 2

    assert transition_job(value, EvaluationState.COMPLETED, now=completed)
    assert value.status == "completed"
    assert value.completed_at == completed
    assert value.current_dimension is None
    assert value.state_version == 3


def test_same_state_is_idempotent_and_terminal_transition_is_rejected() -> None:
    value = job("completed")

    assert not transition_job(value, EvaluationState.COMPLETED)
    assert value.state_version == 1
    with pytest.raises(InvalidJobTransition, match="cannot transition"):
        transition_job(value, EvaluationState.RUNNING)


def test_running_cancel_request_reaches_cancelled_terminal_state() -> None:
    value = job("running")
    requested = datetime(2026, 8, 26, 10, 1, 0)
    cancelled = datetime(2026, 8, 26, 10, 1, 1)

    transition_job(value, "cancel_requested", now=requested)
    transition_job(value, "cancelled", now=cancelled)

    assert value.status == "cancelled"
    assert value.cancel_requested_at == requested
    assert value.cancelled_at == cancelled
    assert value.completed_at == cancelled
    assert value.state_version == 3


def test_failed_and_cancelled_jobs_can_be_requeued() -> None:
    for source in ("failed", "cancelled"):
        value = job(source)
        value.completed_at = datetime(2026, 8, 26, 10, 2, 0)
        transition_job(value, "queued")

        assert value.status == "queued"
        assert value.completed_at is None
        assert value.cancel_requested_at is None
        assert value.cancelled_at is None
