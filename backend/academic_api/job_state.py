from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum


class EvaluationState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {
    EvaluationState.COMPLETED,
    EvaluationState.FAILED,
    EvaluationState.CANCELLED,
}

ALLOWED_TRANSITIONS = {
    EvaluationState.QUEUED: {
        EvaluationState.RUNNING,
        EvaluationState.CANCELLED,
    },
    EvaluationState.RUNNING: {
        EvaluationState.QUEUED,
        EvaluationState.CANCEL_REQUESTED,
        EvaluationState.COMPLETED,
        EvaluationState.FAILED,
    },
    EvaluationState.CANCEL_REQUESTED: {
        EvaluationState.CANCELLED,
        EvaluationState.COMPLETED,
        EvaluationState.FAILED,
    },
    EvaluationState.COMPLETED: set(),
    EvaluationState.FAILED: {EvaluationState.QUEUED},
    EvaluationState.CANCELLED: {EvaluationState.QUEUED},
}


class InvalidJobTransition(ValueError):
    pass


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def transition_job(job, target: EvaluationState | str, *, now=None) -> bool:
    """Apply one validated lifecycle transition to a persisted job object."""
    try:
        source = EvaluationState(job.status)
        target_state = EvaluationState(target)
    except ValueError as exc:
        raise InvalidJobTransition("Unknown evaluation job state.") from exc

    if source == target_state:
        return False
    if target_state not in ALLOWED_TRANSITIONS[source]:
        raise InvalidJobTransition(
            f"Evaluation job cannot transition from {source} to {target_state}."
        )

    timestamp = now or utc_now_naive()
    job.status = target_state.value
    job.state_version = (job.state_version or 0) + 1

    if target_state == EvaluationState.QUEUED:
        job.completed_at = None
        job.cancel_requested_at = None
        job.cancelled_at = None
    elif target_state == EvaluationState.RUNNING:
        if job.started_at is None:
            job.started_at = timestamp
        job.completed_at = None
    elif target_state == EvaluationState.CANCEL_REQUESTED:
        job.cancel_requested_at = timestamp
    elif target_state == EvaluationState.CANCELLED:
        job.cancel_requested_at = job.cancel_requested_at or timestamp
        job.cancelled_at = timestamp
        job.completed_at = timestamp
        job.current_dimension = None
    elif target_state in {EvaluationState.COMPLETED, EvaluationState.FAILED}:
        job.completed_at = timestamp
        if target_state == EvaluationState.COMPLETED:
            job.current_dimension = None

    return True
