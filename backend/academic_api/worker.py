from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from time import perf_counter

from redis.exceptions import RedisError
from sqlalchemy import select

from backend.academic_api.cache.redis_client import (
    EvaluationMessage,
    acknowledge_evaluation,
    close_redis_connection,
    dead_letter_evaluation,
    dequeue_evaluation,
    ensure_evaluation_consumer_group,
)
from backend.academic_api.config import settings
from backend.academic_api.db.session import AsyncSessionLocal
from backend.academic_api.models.evaluation_job import EvaluationJob
from backend.academic_api.models.paper import Paper
from backend.academic_api.job_state import EvaluationState, transition_job
from backend.academic_api.observability import (
    elapsed_milliseconds,
    get_logger,
    log_event,
    reset_correlation_id,
    set_correlation_id,
)
from backend.academic_api.services.evaluation_runner import (
    run_academic_evaluation,
)
from backend.academic_api.public_errors import (
    internal_error_fingerprint,
    public_job_error,
    sanitize_public_payload,
)

DIMENSION_ORDER = (
    "research_question_clarity",
    "methodological_rigor",
    "experimental_design",
    "baseline_quality",
    "result_validity",
    "reproducibility",
    "novelty",
    "limitations_discussion",
    "citation_quality",
)

TOTAL_DIMENSIONS = len(DIMENSION_ORDER)
logger = get_logger("worker")


@dataclass(frozen=True)
class JobProcessResult:
    outcome: str
    error_message: str | None = None


def read_checkpoint_progress(
    checkpoint_path: str | None,
) -> tuple[int, str | None]:
    """
    Read evaluation progress from the checkpoint.

    Returns:
        completed_dimensions,
        current_dimension
    """

    if not checkpoint_path:
        return 0, DIMENSION_ORDER[0]

    path = Path(
        checkpoint_path
    )

    if not path.exists():
        return 0, DIMENSION_ORDER[0]

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        # A partially written or temporarily unavailable checkpoint
        # should never crash the Worker.
        return 0, DIMENSION_ORDER[0]

    raw_results = payload.get(
        "dimension_results",
        [],
    )

    if not isinstance(
        raw_results,
        list,
    ):
        return 0, DIMENSION_ORDER[0]

    completed = set()

    for item in raw_results:
        if not isinstance(
            item,
            dict,
        ):
            continue

        dimension = item.get(
            "dimension"
        )

        if dimension in DIMENSION_ORDER:
            completed.add(
                dimension
            )

    completed_dimensions = len(
        completed
    )

    current_dimension = next(
        (
            dimension
            for dimension in DIMENSION_ORDER
            if dimension not in completed
        ),
        None,
    )

    return (
        completed_dimensions,
        current_dimension,
    )

async def sync_job_progress(
    job_id: str,
) -> None:
    """
    Synchronise checkpoint progress into MySQL.
    """

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EvaluationJob).where(
                EvaluationJob.job_id == job_id
            )
        )

        job = result.scalar_one_or_none()

        if job is None or job.status not in {"running", "cancel_requested"}:
            return

        (
            completed_dimensions,
            current_dimension,
        ) = read_checkpoint_progress(
            job.checkpoint_path
        )

        changed = False

        if (
            job.completed_dimensions
            != completed_dimensions
        ):
            job.completed_dimensions = (
                completed_dimensions
            )
            changed = True

        if (
            job.current_dimension
            != current_dimension
        ):
            job.current_dimension = (
                current_dimension
            )
            changed = True

        if not changed:
            return

        job.state_version = (job.state_version or 0) + 1
        await session.commit()

        print(
            f"[worker] job {job.job_id}: "
            f"progress "
            f"{job.completed_dimensions}/"
            f"{TOTAL_DIMENSIONS}"
            f" current={job.current_dimension}"
        )


async def cancellation_requested(job_id: str) -> bool:
    async with AsyncSessionLocal() as session:
        status_value = await session.scalar(
            select(EvaluationJob.status).where(EvaluationJob.job_id == job_id)
        )
        return status_value in {"cancel_requested", "cancelled"}


async def mark_job_cancelled(job_id: str) -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EvaluationJob)
            .where(EvaluationJob.job_id == job_id)
            .with_for_update()
        )
        job = result.scalar_one_or_none()
        if job is None or job.status == "cancelled":
            return
        if job.status == "running":
            transition_job(job, EvaluationState.CANCEL_REQUESTED)
        if job.status != "cancel_requested":
            return
        transition_job(job, EvaluationState.CANCELLED)
        job.error_code = None
        job.error_message = None
        await session.commit()

async def prepare_job(
    job_id: str,
    *,
    recover_running: bool = False,
) -> dict | None:
    """
    Load the queued job and its paper from MySQL,
    then mark the job as running.

    Only primitive values are returned so that no
    database session stays open during the long
    Ollama evaluation.
    """

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                EvaluationJob,
                Paper,
            )
            .join(
                Paper,
                EvaluationJob.paper_id == Paper.id,
            )
            .where(
                EvaluationJob.job_id == job_id
            )
            .with_for_update()
        )

        row = result.one_or_none()

        if row is None:
            print(
                f"[worker] ignored unknown job message: {job_id}"
            )
            return None

        job, paper = row

        can_claim = job.status == "queued" or (
            recover_running and job.status == "running"
        )

        if not can_claim:
            print(
                f"[worker] ignored duplicate/stale message for "
                f"{job.job_id}: status={job.status}"
            )
            return None

        if job.status == "running":
            print(
                f"[worker] reclaiming stale running job: {job.job_id}"
            )

        managed_pdf_path = job.pdf_path or paper.pdf_path

        if not managed_pdf_path:
            raise ValueError(
                f"Paper {paper.paper_id} has no pdf_path."
            )

        resolved_pdf_path = (
            Path(managed_pdf_path)
            .expanduser()
            .resolve()
        )

        checkpoint_path = job.checkpoint_path

        if not checkpoint_path:
            checkpoint_path = str(
                resolved_pdf_path.with_suffix(
                    ".academic-evaluation-checkpoint.json"
                )
            )

            job.checkpoint_path = checkpoint_path

        transition_job(job, EvaluationState.RUNNING)

        # 清除之前可能留下的错误
        job.error_message = None
        job.error_code = None
        job.result_json = None

        # 从当前 checkpoint 恢复真实进度
        (
            job.completed_dimensions,
            job.current_dimension,
        ) = read_checkpoint_progress(
            checkpoint_path
        )

        await session.commit()

        return {
            "job_id": job.job_id,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "pdf_path": str(resolved_pdf_path),
            "source_url": paper.source_url,
            "model_name": job.model_name,
            "max_chunks": job.max_chunks,
            "checkpoint_path": checkpoint_path,
        }


async def mark_job_completed(
    job_id: str,
    result_json: dict,
) -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EvaluationJob).where(
                EvaluationJob.job_id == job_id
            )
        )

        job = result.scalar_one_or_none()

        if job is None:
            print(
                f"[worker] completed job missing "
                f"from database: {job_id}"
            )
            return

        transition_job(job, EvaluationState.COMPLETED)
        job.result_json = sanitize_public_payload(result_json)
        job.error_message = None
        job.error_code = None

        # 成功完成时强制记录完整 9/9 进度
        job.completed_dimensions = TOTAL_DIMENSIONS
        job.current_dimension = None

        await session.commit()

        print(
            f"[worker] job {job.job_id}: "
            "running -> completed"
        )

        print(
            f"[worker] job {job.job_id}: "
            f"progress "
            f"{job.completed_dimensions}/"
            f"{TOTAL_DIMENSIONS}"
        )


async def mark_job_failed(
    job_id: str,
    error: Exception,
) -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EvaluationJob).where(
                EvaluationJob.job_id == job_id
            )
        )

        job = result.scalar_one_or_none()

        if job is None:
            print(
                f"[worker] failed job missing "
                f"from database: {job_id}"
            )
            return

        # Read the latest checkpoint before marking the job failed.
        (
            completed_dimensions,
            current_dimension,
        ) = read_checkpoint_progress(
            job.checkpoint_path
        )

        transition_job(job, EvaluationState.FAILED)

        # Preserve the real progress reached before the failure.
        job.completed_dimensions = (
            completed_dimensions
        )
        job.current_dimension = (
            current_dimension
        )

        public_error = public_job_error(error)
        job.error_code = public_error.code
        job.error_message = public_error.message

        await session.commit()

        print(
            f"[worker] job {job.job_id}: "
            "running -> failed"
        )

        print(
            f"[worker] job {job.job_id}: "
            f"failed at "
            f"{job.completed_dimensions}/"
            f"{TOTAL_DIMENSIONS}"
            f" current={job.current_dimension}"
        )


async def mark_job_retry_pending(
    job_id: str,
    error: Exception,
) -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EvaluationJob).where(
                EvaluationJob.job_id == job_id
            )
        )
        job = result.scalar_one_or_none()

        if job is None:
            return

        (
            job.completed_dimensions,
            job.current_dimension,
        ) = read_checkpoint_progress(job.checkpoint_path)
        transition_job(job, EvaluationState.QUEUED)
        public_error = public_job_error(error)
        job.error_code = public_error.code
        job.error_message = public_error.message
        job.completed_at = None
        await session.commit()

        print(
            f"[worker] job {job.job_id}: running -> queued for retry"
        )

async def process_job(
    job_id: str,
    *,
    final_attempt: bool = False,
    recover_running: bool = False,
    correlation_id: str | None = None,
) -> JobProcessResult:
    token = set_correlation_id(correlation_id or job_id)
    total_started_at = perf_counter()
    log_event(
        logger,
        "worker.job.received",
        job_id=job_id,
        final_attempt=final_attempt,
        recover_running=recover_running,
    )
    print()
    print("=" * 60)
    print(f"[worker] received job: {job_id}")
    print("=" * 60)

    try:
        prepare_started_at = perf_counter()
        context = await prepare_job(
            job_id,
            recover_running=recover_running,
        )
        log_event(
            logger,
            "worker.stage.completed",
            job_id=job_id,
            stage="prepare",
            duration_ms=elapsed_milliseconds(prepare_started_at),
        )

        if context is None:
            log_event(
                logger,
                "worker.job.skipped",
                job_id=job_id,
                reason="duplicate_or_stale",
            )
            return JobProcessResult(outcome="ack")

        print(
            f"[worker] job {job_id}: "
            "queued -> running"
        )

        print(
            "[worker] paper_id="
            f"{context['paper_id']}"
        )

        print(
            "[worker] model="
            f"{context['model_name']}"
        )

        print(
            "[worker] max_chunks="
            f"{context['max_chunks']}"
        )

        print(
            "[worker] checkpoint="
            f"{context['checkpoint_path']}"
        )

        # -----------------------------------------------------
        # REAL ACADEMIC EVALUATION
        # -----------------------------------------------------

        evaluation_started_at = perf_counter()
        evaluation_task = asyncio.create_task(
            run_academic_evaluation(
                paper_id=context["paper_id"],
                title=context.get("title"),
                pdf_path=context["pdf_path"],
                source_url=context["source_url"],
                checkpoint_path=context["checkpoint_path"],
                model_name=context["model_name"],
                max_chunks=context["max_chunks"],
                hybrid_retrieval_enabled=(
                    settings.academic_hybrid_retrieval_enabled
                ),
                hybrid_candidate_top_k=(
                    settings.academic_hybrid_candidate_top_k
                ),
                hybrid_rrf_k=settings.academic_hybrid_rrf_k,
                embedding_model=settings.academic_embedding_model,
                external_evidence_enabled=(
                    settings.academic_external_evidence_enabled
                ),
                external_evidence_limit=(
                    settings.academic_external_evidence_limit
                ),
            )
        )

        while not evaluation_task.done():
            await asyncio.sleep(2)

            # The evaluation may finish or fail while this task is
            # sleeping. Avoid an unnecessary database call that could
            # hide the original evaluation result or exception.
            if evaluation_task.done():
                break

            if await cancellation_requested(job_id):
                evaluation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await evaluation_task
                await mark_job_cancelled(job_id)
                log_event(logger, "worker.job.cancelled", job_id=job_id)
                return JobProcessResult(outcome="ack")

            await sync_job_progress(
                job_id
            )

        result = await evaluation_task
        log_event(
            logger,
            "worker.stage.completed",
            job_id=job_id,
            stage="evaluate",
            duration_ms=elapsed_milliseconds(evaluation_started_at),
        )

        # Final checkpoint sync after the evaluation task has returned.
        await sync_job_progress(
            job_id
        )

        # Convert the real PaperEvaluation object to JSON.
        # Unit tests may return a plain dict, so support both.
        if hasattr(result, "model_dump"):
            result_json = result.model_dump(
                mode="json"
            )
        elif isinstance(result, dict):
            result_json = result
        else:
            raise TypeError(
                "Academic evaluation returned an "
                f"unsupported result type: {type(result).__name__}"
            )

        # -----------------------------------------------
        # Persist final evaluation
        # -----------------------------------------------

        persistence_started_at = perf_counter()
        await mark_job_completed(
            job_id,
            result_json,
        )
        log_event(
            logger,
            "worker.stage.completed",
            job_id=job_id,
            stage="persist",
            duration_ms=elapsed_milliseconds(persistence_started_at),
        )

        return JobProcessResult(outcome="ack")

    except Exception as exc:
        log_event(
            logger,
            "worker.job.failed",
            level=logging.ERROR,
            job_id=job_id,
            error_type=type(exc).__name__,
            error_fingerprint=internal_error_fingerprint(exc),
            final_attempt=final_attempt,
        )
        print(
            f"[worker] job {job_id} failed: "
            f"{type(exc).__name__} "
            f"fingerprint={internal_error_fingerprint(exc)}"
        )

        safe_error = public_job_error(exc)
        error_message = f"{safe_error.code}: {safe_error.message}"

        try:
            if final_attempt:
                await mark_job_failed(job_id, exc)
            else:
                await mark_job_retry_pending(job_id, exc)
        except Exception as db_exc:
            print(
                "[worker] could not persist retry/failure status: "
                f"{type(db_exc).__name__} "
                "fingerprint="
                f"{internal_error_fingerprint(db_exc)}"
            )
            return JobProcessResult(
                outcome="retry",
                error_message=error_message,
            )

        return JobProcessResult(
            outcome="dead_letter" if final_attempt else "retry",
            error_message=error_message,
        )

    finally:
        log_event(
            logger,
            "worker.job.finished",
            job_id=job_id,
            duration_ms=elapsed_milliseconds(total_started_at),
        )
        reset_correlation_id(token)


async def run_worker() -> None:
    print("=" * 60)
    print("Academic Evaluation Worker")
    print("=" * 60)

    group_ready = False

    print("Waiting for jobs...")
    print("=" * 60)

    try:
        while True:
            try:
                if not group_ready:
                    await ensure_evaluation_consumer_group()
                    group_ready = True

                message = await dequeue_evaluation()

            except RedisError as exc:
                group_ready = False
                print(
                    "[worker] Redis unavailable; retrying: "
                    f"{type(exc).__name__}: {exc}"
                )
                await asyncio.sleep(settings.stream_error_retry_seconds)
                continue

            if message is None:
                continue

            if not message.job_id:
                await dead_letter_evaluation(
                    message,
                    "Stream message has no job_id.",
                )
                continue

            final_attempt = (
                message.delivery_count
                >= settings.stream_max_delivery_attempts
            )

            try:
                process_result = await process_job(
                    message.job_id,
                    final_attempt=final_attempt,
                    recover_running=message.delivery_count > 1,
                    correlation_id=message.correlation_id or message.job_id,
                )

                if process_result.outcome == "ack":
                    await acknowledge_evaluation(message.message_id)
                elif process_result.outcome == "dead_letter":
                    await dead_letter_evaluation(
                        message,
                        process_result.error_message or "Unknown failure",
                    )
                else:
                    print(
                        f"[worker] message {message.message_id} left pending "
                        "for stale recovery"
                    )

            except Exception as exc:
                # Last-resort isolation: one broken job
                # must never terminate the worker process.
                print(
                    "[worker] unexpected processing "
                    f"error for {message.job_id}: {exc}"
                )

    finally:
        await close_redis_connection()


def main() -> None:
    asyncio.run(
        run_worker()
    )


if __name__ == "__main__":
    main()
