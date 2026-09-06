from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.academic_api.db.session import get_db
from backend.academic_api.routers.evaluations import router
from backend.academic_api.security import require_academic_access


class FakeResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def all(self):
        return self.rows

    def one_or_none(self):
        return self.rows[0] if self.rows else None


class ReadSession:
    def __init__(self, rows, total: int = 0) -> None:
        self.rows = rows
        self.total = total

    async def scalar(self, statement):
        return self.total

    async def execute(self, statement):
        return FakeResult(self.rows)


class ActionSession(ReadSession):
    def __init__(self, rows) -> None:
        super().__init__(rows)
        self.added = []
        self.commit_count = 0
        self.events = []

    def add(self, value) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        self.events.append("flush")

    async def refresh(self, value) -> None:
        self.events.append("refresh")

    async def commit(self) -> None:
        self.events.append("commit")
        self.commit_count += 1


def app_for(session: ReadSession) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api")

    async def override_db():
        yield session

    async def allow_access() -> None:
        return None

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[require_academic_access] = allow_access
    return app


def row(*, status: str = "completed", result=None):
    now = datetime(2026, 8, 26, 10, 0, 0)
    job = SimpleNamespace(
        id=7,
        job_id="job-read-1",
        status=status,
        state_version=4,
        retry_count=1,
        completed_dimensions=9 if status == "completed" else 2,
        current_dimension="experimental_design",
        created_at=now,
        updated_at=now,
        started_at=now,
        completed_at=now if status == "completed" else None,
        cancel_requested_at=None,
        cancelled_at=None,
        result_json=result,
        error_code=None,
        error_message=None,
    )
    paper = SimpleNamespace(
        paper_id="arxiv:read-test",
        title="Readable paper",
    )
    return job, paper


def test_paginated_job_listing_returns_metadata_without_results() -> None:
    session = ReadSession([row(result={"overall_score": 90})], total=21)
    with TestClient(app_for(session)) as client:
        response = client.get(
            "/api/academic/evaluations?page=2&page_size=10&job_status=completed"
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["page"] == 2
    assert payload["page_size"] == 10
    assert payload["total"] == 21
    assert payload["total_pages"] == 3
    assert payload["items"][0]["job_id"] == "job-read-1"
    assert "result" not in payload["items"][0]


def test_result_endpoint_returns_sanitized_completed_result() -> None:
    session = ReadSession(
        [
            row(
                result={
                    "overall_score": 90,
                    "checkpoint_path": "/private/job.json",
                }
            )
        ]
    )
    with TestClient(app_for(session)) as client:
        response = client.get(
            "/api/academic/evaluations/job-read-1/result"
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["overall_score"] == 90
    assert "checkpoint_path" not in payload["result"]


def test_job_detail_includes_paper_metadata_and_timestamps() -> None:
    session = ReadSession([row(result={"overall_score": 90})])
    with TestClient(app_for(session)) as client:
        response = client.get("/api/academic/evaluations/job-read-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Readable paper"
    assert payload["created_at"] == "2026-08-26T10:00:00"
    assert payload["updated_at"] == "2026-08-26T10:00:00"


def test_result_endpoint_rejects_non_completed_job() -> None:
    session = ReadSession([row(status="running")])
    with TestClient(app_for(session)) as client:
        response = client.get(
            "/api/academic/evaluations/job-read-1/result"
        )

    assert response.status_code == 409


def test_retry_is_idempotent_and_creates_one_outbox_event() -> None:
    failed_row = row(status="failed")
    failed_row[0].retry_count = 0
    session = ActionSession([failed_row])

    with TestClient(app_for(session)) as client:
        first = client.post("/api/academic/evaluations/job-read-1/retry")
        second = client.post("/api/academic/evaluations/job-read-1/retry")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "queued"
    assert first.json()["retry_count"] == 1
    assert len(session.added) == 1
    assert session.commit_count == 1
    assert session.events.index("refresh") < session.events.index("commit")


def test_cancel_is_idempotent_for_queued_job() -> None:
    queued_row = row(status="running")
    queued_row[0].status = "queued"
    session = ActionSession([queued_row])

    with TestClient(app_for(session)) as client:
        first = client.post("/api/academic/evaluations/job-read-1/cancel")
        second = client.post("/api/academic/evaluations/job-read-1/cancel")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "cancelled"
    assert first.json()["cancelled_at"] is not None
    assert session.commit_count == 1
    assert session.events.index("refresh") < session.events.index("commit")
