from __future__ import annotations

import pytest

from backend.academic_api import worker


@pytest.mark.asyncio
async def test_process_job_completes_successfully(
    monkeypatch,
) -> None:
    job_id = "job-success-001"

    context = {
        "job_id": job_id,
        "paper_id": "arxiv:test-paper",
        "pdf_path": "/tmp/test-paper.pdf",
        "source_url": "https://example.com/test-paper",
        "model_name": "qwen3.5:9b",
        "max_chunks": 4,
        "checkpoint_path": "/tmp/test-checkpoint.json",
    }

    expected_result = {
        "paper_id": "arxiv:test-paper",
        "overall_score": 85.0,
    }

    completed_calls = []
    failed_calls = []

    async def fake_prepare_job(
        received_job_id: str,
        *,
        recover_running: bool = False,
    ) -> dict:
        assert received_job_id == job_id
        assert recover_running is False
        return context

    async def fake_run_academic_evaluation(
        **kwargs,
    ) -> dict:
        assert kwargs["paper_id"] == context["paper_id"]
        assert kwargs["pdf_path"] == context["pdf_path"]
        assert kwargs["model_name"] == context["model_name"]
        assert kwargs["max_chunks"] == context["max_chunks"]
        assert kwargs["hybrid_retrieval_enabled"] == (
            worker.settings.academic_hybrid_retrieval_enabled
        )
        assert kwargs["hybrid_candidate_top_k"] == (
            worker.settings.academic_hybrid_candidate_top_k
        )
        assert kwargs["hybrid_rrf_k"] == worker.settings.academic_hybrid_rrf_k
        assert kwargs["embedding_model"] == (
            worker.settings.academic_embedding_model
        )
        assert kwargs["external_evidence_enabled"] == (
            worker.settings.academic_external_evidence_enabled
        )
        assert kwargs["external_evidence_limit"] == (
            worker.settings.academic_external_evidence_limit
        )
        assert (
            kwargs["checkpoint_path"]
            == context["checkpoint_path"]
        )

        return expected_result

    async def fake_mark_job_completed(
        received_job_id: str,
        result_json: dict,
    ) -> None:
        completed_calls.append(
            (
                received_job_id,
                result_json,
            )
        )

    async def fake_mark_job_failed(
        received_job_id: str,
        error: Exception,
    ) -> None:
        failed_calls.append(
            (
                received_job_id,
                error,
            )
        )

    async def fake_sync_job_progress(
        job_id: str,
    ) -> None:
        return None

    monkeypatch.setattr(
        worker,
        "prepare_job",
        fake_prepare_job,
    )

    monkeypatch.setattr(
        worker,
        "run_academic_evaluation",
        fake_run_academic_evaluation,
    )

    monkeypatch.setattr(
        worker,
        "mark_job_completed",
        fake_mark_job_completed,
    )

    monkeypatch.setattr(
        worker,
        "mark_job_failed",
        fake_mark_job_failed,
    )

    monkeypatch.setattr(
        worker,
        "sync_job_progress",
        fake_sync_job_progress,
    )

    result = await worker.process_job(job_id)

    assert completed_calls == [
        (
            job_id,
            expected_result,
        )
    ]

    assert failed_calls == []
    assert result.outcome == "ack"


@pytest.mark.asyncio
async def test_process_job_retries_then_dead_letters_evaluation_failure(
    monkeypatch,
) -> None:
    job_id = "job-failed-001"

    context = {
        "job_id": job_id,
        "paper_id": "arxiv:test-paper",
        "pdf_path": "/tmp/test-paper.pdf",
        "source_url": None,
        "model_name": "qwen3.5:9b",
        "max_chunks": 4,
        "checkpoint_path": "/tmp/test-checkpoint.json",
    }

    completed_calls = []
    failed_calls = []
    retry_calls = []

    async def fake_prepare_job(
        received_job_id: str,
        *,
        recover_running: bool = False,
    ) -> dict:
        assert received_job_id == job_id
        return context

    async def fake_run_academic_evaluation(
        **kwargs,
    ) -> dict:
        raise RuntimeError(
            "simulated evaluation failure"
        )

    async def fake_mark_job_completed(
        received_job_id: str,
        result_json: dict,
    ) -> None:
        completed_calls.append(
            (
                received_job_id,
                result_json,
            )
        )

    async def fake_mark_job_failed(
        received_job_id: str,
        error: Exception,
    ) -> None:
        failed_calls.append(
            (
                received_job_id,
                error,
            )
        )

    async def fake_mark_job_retry_pending(
        received_job_id: str,
        error: Exception,
    ) -> None:
        retry_calls.append((received_job_id, error))

    monkeypatch.setattr(
        worker,
        "prepare_job",
        fake_prepare_job,
    )

    monkeypatch.setattr(
        worker,
        "run_academic_evaluation",
        fake_run_academic_evaluation,
    )

    monkeypatch.setattr(
        worker,
        "mark_job_completed",
        fake_mark_job_completed,
    )

    monkeypatch.setattr(
        worker,
        "mark_job_failed",
        fake_mark_job_failed,
    )
    monkeypatch.setattr(
        worker,
        "mark_job_retry_pending",
        fake_mark_job_retry_pending,
    )

    retry_result = await worker.process_job(job_id)

    assert completed_calls == []
    assert failed_calls == []
    assert len(retry_calls) == 1
    assert retry_result.outcome == "retry"
    assert retry_result.error_message == (
        "evaluation_failed: The academic evaluation failed."
    )

    dead_letter_result = await worker.process_job(
        job_id,
        final_attempt=True,
    )

    assert len(failed_calls) == 1

    failed_job_id, error = failed_calls[0]

    assert failed_job_id == job_id
    assert isinstance(
        error,
        RuntimeError,
    )
    assert (
        str(error)
        == "simulated evaluation failure"
    )
    assert dead_letter_result.outcome == "dead_letter"


@pytest.mark.asyncio
async def test_process_job_ignores_duplicate_or_stale_message(
    monkeypatch,
) -> None:
    calls = []

    async def fake_prepare_job(
        job_id: str,
        *,
        recover_running: bool = False,
    ) -> None:
        calls.append(("prepare", job_id))
        return None

    async def unexpected_evaluation(**kwargs):
        calls.append(("evaluate", kwargs))

    monkeypatch.setattr(worker, "prepare_job", fake_prepare_job)
    monkeypatch.setattr(
        worker,
        "run_academic_evaluation",
        unexpected_evaluation,
    )

    result = await worker.process_job("duplicate-job")

    assert calls == [("prepare", "duplicate-job")]
    assert result.outcome == "ack"


@pytest.mark.asyncio
async def test_process_job_cooperatively_cancels_running_evaluation(
    monkeypatch,
) -> None:
    job_id = "job-cancel-001"
    context = {
        "job_id": job_id,
        "paper_id": "arxiv:cancel-test",
        "pdf_path": "/tmp/test.pdf",
        "source_url": None,
        "model_name": "qwen3.5:9b",
        "max_chunks": 1,
        "checkpoint_path": "/tmp/test.json",
    }
    cancelled = []
    original_sleep = worker.asyncio.sleep

    async def fake_prepare(received_job_id, *, recover_running=False):
        return context

    async def long_evaluation(**kwargs):
        await original_sleep(60)

    async def immediate_sleep(seconds):
        await original_sleep(0)

    async def cancel_requested(received_job_id):
        return True

    async def mark_cancelled(received_job_id):
        cancelled.append(received_job_id)

    monkeypatch.setattr(worker, "prepare_job", fake_prepare)
    monkeypatch.setattr(worker, "run_academic_evaluation", long_evaluation)
    monkeypatch.setattr(worker.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(worker, "cancellation_requested", cancel_requested)
    monkeypatch.setattr(worker, "mark_job_cancelled", mark_cancelled)

    result = await worker.process_job(job_id)

    assert result.outcome == "ack"
    assert cancelled == [job_id]
