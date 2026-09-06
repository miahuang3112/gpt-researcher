from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.academic_api.config import settings
from backend.academic_api.db.session import AsyncSessionLocal
from backend.academic_api.models.evaluation_job import EvaluationJob
from backend.academic_api.models.paper import Paper


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class ActiveJobDeletionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetentionPolicy:
    pdf_days: int = 30
    checkpoint_days: int = 7
    result_days: int = 90

    def __post_init__(self) -> None:
        if min(self.pdf_days, self.checkpoint_days, self.result_days) < 0:
            raise ValueError("Retention periods cannot be negative.")


@dataclass
class PurgeCounts:
    pdfs: int = 0
    checkpoints: int = 0
    results: int = 0


def delete_managed_file(path_value: str | None, *, root: str | Path) -> bool:
    """Delete only a regular file that resolves inside its managed root."""
    if not path_value:
        return False

    managed_root = Path(root).expanduser().resolve()
    path = Path(path_value).expanduser().resolve()

    if not path.is_relative_to(managed_root):
        return False
    if not path.is_file():
        return False

    path.unlink()
    return True


def purge_expired_job_artifacts(
    job,
    paper,
    *,
    now: datetime,
    policy: RetentionPolicy,
    pdf_storage_root: str | Path,
    checkpoint_storage_root: str | Path,
    allow_pdf_delete: bool = True,
) -> PurgeCounts:
    """Apply terminal-job retention without deleting the audit row itself."""
    counts = PurgeCounts()
    if job.status not in TERMINAL_STATUSES or job.completed_at is None:
        return counts

    completed_at = job.completed_at
    if completed_at.tzinfo is None:
        completed_at = completed_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if completed_at <= now - timedelta(days=policy.checkpoint_days):
        if delete_managed_file(
            job.checkpoint_path,
            root=checkpoint_storage_root,
        ):
            counts.checkpoints += 1
        job.checkpoint_path = None

    if completed_at <= now - timedelta(days=policy.pdf_days):
        pdf_path = job.pdf_path
        if allow_pdf_delete and delete_managed_file(
            pdf_path,
            root=pdf_storage_root,
        ):
            counts.pdfs += 1
        job.pdf_path = None
        if (
            allow_pdf_delete
            and paper is not None
            and paper.pdf_path == pdf_path
        ):
            paper.pdf_path = None

    if completed_at <= now - timedelta(days=policy.result_days):
        if job.result_json is not None:
            counts.results += 1
        job.result_json = None

    return counts


async def purge_expired_artifacts(
    db: AsyncSession,
    *,
    now: datetime | None = None,
    policy: RetentionPolicy | None = None,
    pdf_storage_root: str | Path | None = None,
    checkpoint_storage_root: str | Path | None = None,
) -> PurgeCounts:
    policy = policy or RetentionPolicy(
        pdf_days=settings.academic_pdf_retention_days,
        checkpoint_days=settings.academic_checkpoint_retention_days,
        result_days=settings.academic_result_retention_days,
    )
    now = now or datetime.now(timezone.utc)
    result = await db.execute(
        select(EvaluationJob, Paper).join(
            Paper,
            EvaluationJob.paper_id == Paper.id,
        )
    )
    totals = PurgeCounts()

    for job, paper in result.all():
        other_pdf_references = 0
        if job.pdf_path:
            other_pdf_references = int(
                await db.scalar(
                    select(func.count(EvaluationJob.id)).where(
                        EvaluationJob.pdf_path == job.pdf_path,
                        EvaluationJob.id != job.id,
                    )
                )
                or 0
            )
        counts = purge_expired_job_artifacts(
            job,
            paper,
            now=now,
            policy=policy,
            pdf_storage_root=(
                pdf_storage_root or settings.academic_pdf_storage_dir
            ),
            checkpoint_storage_root=(
                checkpoint_storage_root
                or settings.academic_checkpoint_storage_dir
            ),
            allow_pdf_delete=other_pdf_references == 0,
        )
        totals.pdfs += counts.pdfs
        totals.checkpoints += counts.checkpoints
        totals.results += counts.results

    await db.commit()
    return totals


async def delete_evaluation_job(
    db: AsyncSession,
    *,
    job_id: str,
    pdf_storage_root: str | Path,
    checkpoint_storage_root: str | Path,
) -> bool:
    result = await db.execute(
        select(EvaluationJob, Paper)
        .join(Paper, EvaluationJob.paper_id == Paper.id)
        .where(EvaluationJob.job_id == job_id)
        .with_for_update()
    )
    row = result.one_or_none()
    if row is None:
        return False

    job, paper = row
    if job.status not in TERMINAL_STATUSES:
        raise ActiveJobDeletionError(job_id)

    pdf_path = job.pdf_path
    checkpoint_path = job.checkpoint_path
    other_pdf_references = 0
    if pdf_path:
        other_pdf_references = int(
            await db.scalar(
                select(func.count(EvaluationJob.id)).where(
                    EvaluationJob.pdf_path == pdf_path,
                    EvaluationJob.id != job.id,
                )
            )
            or 0
        )
    remaining = await db.scalar(
        select(func.count(EvaluationJob.id)).where(
            EvaluationJob.paper_id == paper.id,
            EvaluationJob.id != job.id,
        )
    )
    await db.delete(job)
    if not remaining:
        await db.delete(paper)
    await db.commit()

    if other_pdf_references == 0:
        delete_managed_file(pdf_path, root=pdf_storage_root)
    delete_managed_file(checkpoint_path, root=checkpoint_storage_root)
    return True


async def _run_retention() -> None:
    async with AsyncSessionLocal() as db:
        await purge_expired_artifacts(db)


def main() -> None:
    asyncio.run(_run_retention())


if __name__ == "__main__":
    main()
