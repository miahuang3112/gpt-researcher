from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest


API_BASE_URL = os.getenv(
    "ACADEMIC_API_BASE_URL",
    "http://127.0.0.1:8001",
)

E2E_PDF_PATH = os.getenv(
    "ACADEMIC_E2E_PDF_PATH",
    "data/sample.pdf",
)

E2E_PAPER_ID = os.getenv(
    "ACADEMIC_E2E_PAPER_ID",
    "arxiv:2410.02902",
)

E2E_MODEL = os.getenv(
    "ACADEMIC_E2E_MODEL",
    "qwen3.5:9b",
)

E2E_MAX_CHUNKS = int(
    os.getenv(
        "ACADEMIC_E2E_MAX_CHUNKS",
        "4",
    )
)

E2E_TIMEOUT_SECONDS = float(
    os.getenv(
        "ACADEMIC_E2E_TIMEOUT_SECONDS",
        "60",
    )
)
E2E_QUEUE_TIMEOUT_SECONDS = float(
    os.getenv("ACADEMIC_E2E_QUEUE_TIMEOUT_SECONDS", str(E2E_TIMEOUT_SECONDS))
)
E2E_RUN_TIMEOUT_SECONDS = float(
    os.getenv("ACADEMIC_E2E_RUN_TIMEOUT_SECONDS", str(E2E_TIMEOUT_SECONDS))
)


async def wait_for_terminal_status(
    client: httpx.AsyncClient,
    job_id: str,
    *,
    queue_timeout_seconds: float = 60.0,
    run_timeout_seconds: float = 60.0,
) -> dict:
    """
    Poll the evaluation API until the job reaches
    completed or failed.
    """

    loop = asyncio.get_running_loop()
    queue_deadline = loop.time() + queue_timeout_seconds
    running_deadline: float | None = None

    last_payload: dict | None = None

    while True:
        response = await client.get(
            f"/api/academic/evaluations/{job_id}"
        )

        assert response.status_code == 200

        payload = response.json()
        last_payload = payload

        status = payload["status"]

        if status not in {"queued", "pending"} and running_deadline is None:
            running_deadline = loop.time() + run_timeout_seconds

        print(
            f"[e2e] job={job_id} "
            f"status={status}"
        )

        if status in {
            "completed",
            "failed",
        }:
            return payload

        active_deadline = running_deadline or queue_deadline
        if loop.time() >= active_deadline:
            phase = "running" if running_deadline is not None else "queue"
            phase_timeout = (
                run_timeout_seconds if running_deadline is not None else queue_timeout_seconds
            )
            raise AssertionError(
                f"Evaluation job exceeded the {phase} timeout of "
                f"{phase_timeout} seconds. Last payload: {last_payload}"
            )

        await asyncio.sleep(0.5)

@pytest.mark.asyncio
async def test_academic_evaluation_full_e2e() -> None:
    """
    Real integration test:

    FastAPI
        -> MySQL
        -> Redis
        -> Worker
        -> Academic evaluation pipeline
        -> MySQL
        -> GET evaluation result

    This test is disabled by default because it
    requires the local infrastructure to be running.
    """

    if os.getenv(
        "RUN_ACADEMIC_E2E"
    ) != "1":
        pytest.skip(
            "Set RUN_ACADEMIC_E2E=1 "
            "to run the real academic E2E test."
        )

    pdf_path = Path(
        E2E_PDF_PATH
    ).expanduser().resolve()

    assert pdf_path.exists(), (
        f"E2E PDF does not exist: "
        f"{pdf_path}"
    )

    assert pdf_path.is_file(), (
        f"E2E PDF path is not a file: "
        f"{pdf_path}"
    )

    async with httpx.AsyncClient(
        base_url=API_BASE_URL,
        timeout=30.0,
    ) as client:

        # --------------------------------------------------
        # 1. Health check
        # --------------------------------------------------

        health_response = await client.get(
            "/health"
        )

        assert (
            health_response.status_code
            == 200
        )

        health_payload = (
            health_response.json()
        )

        assert (
            health_payload["status"]
            == "ok"
        )

        # --------------------------------------------------
        # 2. Submit evaluation
        # --------------------------------------------------

        create_response = await client.post(
            "/api/academic/evaluations",
            json={
                "paper_id": E2E_PAPER_ID,
                "pdf_path": str(
                    pdf_path
                ),
                "model_name": E2E_MODEL,
                "max_chunks": (
                    E2E_MAX_CHUNKS
                ),
            },
        )

        assert create_response.status_code in {
            200,
            201,
            202,
        }, create_response.text

        create_payload = (
            create_response.json()
        )

        print(
            "[e2e] POST response:",
            create_payload,
        )

        job_id = create_payload[
            "job_id"
        ]

        assert job_id
        assert (
            create_payload["status"]
            == "queued"
        )

        # --------------------------------------------------
        # 3. Poll GET endpoint
        # --------------------------------------------------

        final_payload = (
            await wait_for_terminal_status(
                client,
                job_id,
                queue_timeout_seconds=E2E_QUEUE_TIMEOUT_SECONDS,
                run_timeout_seconds=E2E_RUN_TIMEOUT_SECONDS,
            )
        )

        # --------------------------------------------------
        # 4. Final assertions
        # --------------------------------------------------

        assert (
            final_payload["job_id"]
            == job_id
        )

        assert (
            final_payload["status"]
            == "completed"
        ), final_payload

        assert (
            final_payload.get(
                "error_message"
            )
            is None
        )

        result = (
            final_payload.get(
                "result"
            )
        )

        assert isinstance(
            result,
            dict,
        )

        assert result

        assert (
            result.get(
                "worker_smoke_test"
            )
            is not True
        )

        print(
            "[e2e] completed job:",
            job_id,
        )
