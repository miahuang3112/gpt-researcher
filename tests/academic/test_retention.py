from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.academic_api.services.retention import (
    RetentionPolicy,
    delete_managed_file,
    purge_expired_job_artifacts,
)


def test_delete_managed_file_never_deletes_outside_root(tmp_path: Path) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"private")

    assert not delete_managed_file(str(outside), root=managed)
    assert outside.exists()


def test_terminal_job_artifacts_expire_independently(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdfs"
    checkpoint_root = tmp_path / "checkpoints"
    pdf_root.mkdir()
    checkpoint_root.mkdir()
    pdf = pdf_root / "random.pdf"
    checkpoint = checkpoint_root / "job.json"
    pdf.write_bytes(b"pdf")
    checkpoint.write_text("{}", encoding="utf-8")
    now = datetime.now(timezone.utc)
    job = SimpleNamespace(
        status="completed",
        completed_at=now - timedelta(days=40),
        pdf_path=str(pdf),
        checkpoint_path=str(checkpoint),
        result_json={"score": 80},
    )
    paper = SimpleNamespace(pdf_path=str(pdf))

    counts = purge_expired_job_artifacts(
        job,
        paper,
        now=now,
        policy=RetentionPolicy(
            pdf_days=30,
            checkpoint_days=7,
            result_days=90,
        ),
        pdf_storage_root=pdf_root,
        checkpoint_storage_root=checkpoint_root,
    )

    assert counts.pdfs == 1
    assert counts.checkpoints == 1
    assert counts.results == 0
    assert job.pdf_path is None
    assert job.checkpoint_path is None
    assert job.result_json == {"score": 80}
    assert paper.pdf_path is None


def test_result_expires_but_minimal_audit_state_remains(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    job = SimpleNamespace(
        status="failed",
        completed_at=now - timedelta(days=100),
        pdf_path=None,
        checkpoint_path=None,
        result_json={"private": "result"},
    )

    counts = purge_expired_job_artifacts(
        job,
        None,
        now=now,
        policy=RetentionPolicy(30, 7, 90),
        pdf_storage_root=tmp_path / "pdfs",
        checkpoint_storage_root=tmp_path / "checkpoints",
    )

    assert counts.results == 1
    assert job.result_json is None
    assert job.status == "failed"


def test_active_job_is_never_purged(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    job = SimpleNamespace(
        status="running",
        completed_at=now - timedelta(days=365),
        pdf_path="/managed/paper.pdf",
        checkpoint_path="/managed/job.json",
        result_json={"partial": True},
    )

    counts = purge_expired_job_artifacts(
        job,
        None,
        now=now,
        policy=RetentionPolicy(0, 0, 0),
        pdf_storage_root=tmp_path,
        checkpoint_storage_root=tmp_path,
    )

    assert counts.pdfs == counts.checkpoints == counts.results == 0
    assert job.result_json == {"partial": True}


def test_negative_retention_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        RetentionPolicy(pdf_days=-1)
