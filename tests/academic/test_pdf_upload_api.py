from __future__ import annotations

from pathlib import Path
from datetime import datetime

import fitz
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.academic_api.config import settings
from backend.academic_api.db.session import get_db
from backend.academic_api.models.evaluation_job import EvaluationJob
from backend.academic_api.models.outbox_event import OutboxEvent
from backend.academic_api.models.paper import Paper
from backend.academic_api.routers.evaluations import router
from backend.academic_api.security import require_academic_access


def pdf_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "browser-paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Secure browser upload")
    document.save(str(path))
    document.close()
    return path.read_bytes()


class FakeSession:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.fail_commit = fail_commit
        self.added: list[object] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.events: list[str] = []

    async def scalar(self, statement):
        return None

    def add(self, value: object) -> None:
        self.added.append(value)
        if isinstance(value, Paper):
            value.id = 1
        elif isinstance(value, EvaluationJob):
            value.id = 2

    async def flush(self) -> None:
        self.events.append("flush")
        return None

    async def refresh(self, value: object, attribute_names=None) -> None:
        self.events.append("refresh")
        if isinstance(value, EvaluationJob):
            value.created_at = datetime(2026, 8, 26, 12, 0, 0)
            value.updated_at = datetime(2026, 8, 26, 12, 0, 0)

    async def commit(self) -> None:
        self.events.append("commit")
        if self.fail_commit:
            raise RuntimeError("database unavailable")
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


def create_app(session: FakeSession) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api")

    async def override_db():
        yield session

    async def allow_access() -> None:
        return None

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[require_academic_access] = allow_access
    return app


def test_secure_upload_creates_job_and_outbox(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = tmp_path / "managed-pdfs"
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setattr(settings, "academic_pdf_storage_dir", str(storage))
    monkeypatch.setattr(
        settings,
        "academic_checkpoint_storage_dir",
        str(checkpoints),
    )
    session = FakeSession()

    with TestClient(create_app(session)) as client:
        response = client.post(
            "/api/academic/evaluations/upload",
            data={
                "paper_id": "arxiv:web-upload-test",
                "title": "Uploaded paper",
                "model_name": "qwen3.5:9b",
                "max_chunks": "2",
            },
            files={
                "file": (
                    "../../user-controlled.pdf",
                    pdf_bytes(tmp_path),
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["paper_id"] == "arxiv:web-upload-test"
    assert payload["status"] == "queued"
    assert payload["created_at"] == "2026-08-26T12:00:00"
    assert payload["updated_at"] == "2026-08-26T12:00:00"
    assert "pdf_path" not in payload
    assert "checkpoint_path" not in payload
    assert session.commit_count == 1
    assert session.events.index("refresh") < session.events.index("commit")
    assert any(isinstance(value, Paper) for value in session.added)
    assert any(isinstance(value, EvaluationJob) for value in session.added)
    assert any(isinstance(value, OutboxEvent) for value in session.added)
    stored_files = list(storage.glob("*.pdf"))
    assert len(stored_files) == 1
    assert stored_files[0].name != "user-controlled.pdf"
    assert stored_files[0].stat().st_mode & 0o077 == 0


def test_secure_upload_rejects_non_pdf_before_database_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = tmp_path / "managed-pdfs"
    monkeypatch.setattr(settings, "academic_pdf_storage_dir", str(storage))
    session = FakeSession()

    with TestClient(create_app(session)) as client:
        response = client.post(
            "/api/academic/evaluations/upload",
            data={"paper_id": "invalid-upload"},
            files={"file": ("paper.txt", b"not a pdf", "text/plain")},
        )

    assert response.status_code == 422
    assert session.added == []
    assert not storage.exists() or list(storage.iterdir()) == []


def test_secure_upload_removes_file_when_database_commit_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = tmp_path / "managed-pdfs"
    monkeypatch.setattr(settings, "academic_pdf_storage_dir", str(storage))
    session = FakeSession(fail_commit=True)

    with TestClient(create_app(session), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/academic/evaluations/upload",
            data={"paper_id": "failed-upload"},
            files={
                "file": (
                    "paper.pdf",
                    pdf_bytes(tmp_path),
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 500
    assert session.rollback_count == 1
    assert list(storage.iterdir()) == []
