from pathlib import Path
from typing import Annotated
from uuid import uuid4
from datetime import datetime, timezone
import json
from time import perf_counter
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.academic_api.db.session import AsyncSessionLocal, get_db
from backend.academic_api.job_state import EvaluationState, transition_job
from backend.academic_api.models.evaluation_job import EvaluationJob
from backend.academic_api.models.paper import Paper
from backend.academic_api.observability import (
    elapsed_milliseconds,
    get_correlation_id,
    get_logger,
    log_event,
)
from backend.academic_api.schemas import (
    CreateEvaluationRequest,
    EvaluationJobListResponse,
    EvaluationJobResponse,
    EvaluationJobStatus,
    EvaluationJobSummary,
    EvaluationResultResponse,
)
from backend.academic_api.services.outbox import (
    build_evaluation_requested_event,
)
from backend.academic_api.services.job_events import evaluation_event_stream
from backend.academic_api.security import require_academic_access
from backend.academic_api.config import settings
from backend.academic_api.services.secure_pdf import (
    PdfSecurityError,
    StoredPdf,
    store_validated_pdf,
    store_validated_pdf_upload,
)
from backend.academic_api.public_errors import sanitize_public_payload
from backend.academic_api.services.retention import (
    ActiveJobDeletionError,
    delete_evaluation_job,
)
from backend.academic_api.services.result_export import (
    build_result_export,
    render_result_pdf,
)

TOTAL_DIMENSIONS = 9
logger = get_logger("evaluations")


def _public_error_fields(job: EvaluationJob) -> tuple[str | None, str | None]:
    if job.status == "failed" and not job.error_code:
        return "evaluation_failed", "The academic evaluation failed."
    return job.error_code, sanitize_public_payload(job.error_message)

router = APIRouter(
    prefix="/academic/evaluations",
    tags=["academic-evaluations"],
    dependencies=[Security(require_academic_access)],
)


def _create_job_response(
    job: EvaluationJob,
    paper: Paper,
) -> EvaluationJobResponse:
    completed_dimensions = job.completed_dimensions or 0
    if job.status == "completed":
        completed_dimensions = TOTAL_DIMENSIONS
    completed_dimensions = max(
        0,
        min(completed_dimensions, TOTAL_DIMENSIONS),
    )
    duration_seconds: float | None = None
    if job.started_at is not None:
        end_time = job.completed_at
        if end_time is None:
            end_time = datetime.now(timezone.utc).replace(tzinfo=None)
        duration_seconds = max(
            0.0,
            (end_time - job.started_at).total_seconds(),
        )
    error_code, error_message = _public_error_fields(job)
    return EvaluationJobResponse(
        job_id=job.job_id,
        paper_id=paper.paper_id,
        title=paper.title,
        status=job.status,
        state_version=job.state_version or 1,
        retry_count=job.retry_count or 0,
        completed_dimensions=completed_dimensions,
        total_dimensions=TOTAL_DIMENSIONS,
        progress=round(completed_dimensions / TOTAL_DIMENSIONS, 4),
        current_dimension=job.current_dimension,
        started_at=job.started_at,
        completed_at=job.completed_at,
        cancel_requested_at=job.cancel_requested_at,
        cancelled_at=job.cancelled_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        duration_seconds=duration_seconds,
        result=sanitize_public_payload(job.result_json),
        error_code=error_code,
        error_message=error_message,
    )


async def _persist_evaluation(
    payload: CreateEvaluationRequest,
    stored_pdf: StoredPdf,
    db: AsyncSession,
    *,
    started_at: float,
) -> EvaluationJobResponse:
    public_job_id = uuid4().hex
    paper_id = (payload.paper_id or "").strip() or f"local:{uuid4().hex}"
    title = (payload.title or "").strip() or None
    source_url = (payload.source_url or "").strip() or None
    try:
        paper = await db.scalar(
            select(Paper).where(Paper.paper_id == paper_id)
        )
        if paper is None:
            paper = Paper(
                paper_id=paper_id,
                title=title,
                source_url=source_url,
                pdf_path=str(stored_pdf.path),
            )
            db.add(paper)
            await db.flush()
        else:
            paper.pdf_path = str(stored_pdf.path)
            if title is not None:
                paper.title = title
            if source_url is not None:
                paper.source_url = source_url

        job = EvaluationJob(
            job_id=public_job_id,
            paper_id=paper.id,
            status="queued",
            checkpoint_path=str(
                Path(settings.academic_checkpoint_storage_dir)
                / f"{public_job_id}.json"
            ),
            pdf_path=str(stored_pdf.path),
            model_name=payload.model_name,
            max_chunks=payload.max_chunks,
        )
        db.add(job)
        await db.flush()
        db.add(
            build_evaluation_requested_event(
                job_database_id=job.id,
                public_job_id=job.job_id,
                correlation_id=get_correlation_id(),
            )
        )
        await db.flush()
        await db.refresh(
            job,
            attribute_names=[
                "created_at",
                "updated_at",
                "state_version",
                "retry_count",
            ],
        )
        response = _create_job_response(job, paper)
        await db.commit()
    except Exception:
        await db.rollback()
        stored_pdf.path.unlink(missing_ok=True)
        raise

    log_event(
        logger,
        "evaluation_job.created",
        job_id=job.job_id,
        paper_id=paper.paper_id,
        status=job.status,
        state_version=job.state_version or 1,
        retry_count=job.retry_count or 0,
        duration_ms=elapsed_milliseconds(started_at),
    )
    return response


@router.post(
    "",
    response_model=EvaluationJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_evaluation(
    payload: CreateEvaluationRequest,
    db: AsyncSession = Depends(get_db),
) -> EvaluationJobResponse:
    started_at = perf_counter()
    try:
        stored_pdf = await run_in_threadpool(
            store_validated_pdf,
            payload.pdf_path,
            allowed_import_roots=settings.pdf_import_roots,
            storage_dir=settings.academic_pdf_storage_dir,
            max_bytes=settings.academic_pdf_max_bytes,
            max_pages=settings.academic_pdf_max_pages,
            max_extracted_characters=(
                settings.academic_pdf_max_extracted_characters
            ),
        )
    except PdfSecurityError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return await _persist_evaluation(
        payload,
        stored_pdf,
        db,
        started_at=started_at,
    )


@router.post(
    "/upload",
    response_model=EvaluationJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_evaluation(
    file: Annotated[UploadFile, File(description="PDF document to evaluate")],
    paper_id: Annotated[str | None, Form(min_length=1, max_length=255)] = None,
    title: Annotated[str | None, Form(max_length=500)] = None,
    source_url: Annotated[str | None, Form(max_length=1000)] = None,
    model_name: Annotated[str, Form(min_length=1, max_length=255)] = "qwen3.5:9b",
    max_chunks: Annotated[int, Form(ge=1, le=20)] = 4,
    db: AsyncSession = Depends(get_db),
) -> EvaluationJobResponse:
    """Securely upload a PDF and atomically create its evaluation job."""
    started_at = perf_counter()
    try:
        stored_pdf = await run_in_threadpool(
            store_validated_pdf_upload,
            file.file,
            content_type=file.content_type,
            storage_dir=settings.academic_pdf_storage_dir,
            max_bytes=settings.academic_pdf_max_bytes,
            max_pages=settings.academic_pdf_max_pages,
            max_extracted_characters=(
                settings.academic_pdf_max_extracted_characters
            ),
        )
    except PdfSecurityError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()

    payload = CreateEvaluationRequest(
        paper_id=paper_id,
        pdf_path=str(stored_pdf.path),
        title=title,
        source_url=source_url,
        model_name=model_name,
        max_chunks=max_chunks,
    )
    return await _persist_evaluation(
        payload,
        stored_pdf,
        db,
        started_at=started_at,
    )


@router.get("", response_model=EvaluationJobListResponse)
async def list_evaluations(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    job_status: EvaluationJobStatus | None = None,
    db: AsyncSession = Depends(get_db),
) -> EvaluationJobListResponse:
    filters = []
    if job_status is not None:
        filters.append(EvaluationJob.status == job_status.value)

    total = int(
        await db.scalar(
            select(func.count(EvaluationJob.id)).where(*filters)
        )
        or 0
    )
    result = await db.execute(
        select(EvaluationJob, Paper)
        .join(Paper, EvaluationJob.paper_id == Paper.id)
        .where(*filters)
        .order_by(EvaluationJob.created_at.desc(), EvaluationJob.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = []
    for job, paper in result.all():
        completed = TOTAL_DIMENSIONS if job.status == "completed" else max(
            0,
            min(job.completed_dimensions or 0, TOTAL_DIMENSIONS),
        )
        items.append(
            EvaluationJobSummary(
                job_id=job.job_id,
                paper_id=paper.paper_id,
                title=paper.title,
                status=job.status,
                state_version=job.state_version or 1,
                retry_count=job.retry_count or 0,
                completed_dimensions=completed,
                total_dimensions=TOTAL_DIMENSIONS,
                progress=round(completed / TOTAL_DIMENSIONS, 4),
                current_dimension=(
                    job.current_dimension if job.status == "running" else None
                ),
                created_at=job.created_at,
                updated_at=job.updated_at,
                completed_at=job.completed_at,
            )
        )

    total_pages = (total + page_size - 1) // page_size if total else 0
    return EvaluationJobListResponse(
        items=items,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
    )

@router.get(
    "/{job_id}",
    response_model=EvaluationJobResponse,
)
async def get_evaluation(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvaluationJobResponse:
    result = await db.execute(
        select(EvaluationJob, Paper)
        .join(
            Paper,
            EvaluationJob.paper_id == Paper.id,
        )
        .where(
            EvaluationJob.job_id == job_id,
        )
    )

    row = result.one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation job not found.",
        )

    job, paper = row
    return _create_job_response(job, paper)


@router.get(
    "/{job_id}/result",
    response_model=EvaluationResultResponse,
)
async def get_evaluation_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvaluationResultResponse:
    result = await db.execute(
        select(EvaluationJob, Paper)
        .join(Paper, EvaluationJob.paper_id == Paper.id)
        .where(EvaluationJob.job_id == job_id)
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation job not found.",
        )
    job, paper = row
    if job.status != "completed" or job.result_json is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Evaluation result is not available.",
        )
    return EvaluationResultResponse(
        job_id=job.job_id,
        paper_id=paper.paper_id,
        status=job.status,
        completed_at=job.completed_at,
        result=sanitize_public_payload(job.result_json),
    )


async def _locked_job_with_paper(
    db: AsyncSession,
    job_id: str,
) -> tuple[EvaluationJob, Paper]:
    result = await db.execute(
        select(EvaluationJob, Paper)
        .join(Paper, EvaluationJob.paper_id == Paper.id)
        .where(EvaluationJob.job_id == job_id)
        .with_for_update()
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation job not found.",
        )
    return row


@router.post("/{job_id}/retry", response_model=EvaluationJobResponse)
async def retry_evaluation(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvaluationJobResponse:
    job, paper = await _locked_job_with_paper(db, job_id)

    if job.status in {"queued", "running", "cancel_requested"}:
        if (job.retry_count or 0) > 0:
            return _create_job_response(job, paper)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only failed or cancelled evaluations can be retried.",
        )
    if job.status not in {"failed", "cancelled"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only failed or cancelled evaluations can be retried.",
        )

    transition_job(job, EvaluationState.QUEUED)
    job.retry_count = (job.retry_count or 0) + 1
    job.error_code = None
    job.error_message = None
    job.result_json = None
    db.add(
        build_evaluation_requested_event(
            job_database_id=job.id,
            public_job_id=job.job_id,
            correlation_id=get_correlation_id(),
        )
    )
    await db.flush()
    await db.refresh(job)
    response = _create_job_response(job, paper)
    await db.commit()
    return response


@router.post("/{job_id}/cancel", response_model=EvaluationJobResponse)
async def cancel_evaluation(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvaluationJobResponse:
    job, paper = await _locked_job_with_paper(db, job_id)

    if job.status in {"cancel_requested", "cancelled"}:
        return _create_job_response(job, paper)
    if job.status == "queued":
        transition_job(job, EvaluationState.CANCELLED)
    elif job.status == "running":
        transition_job(job, EvaluationState.CANCEL_REQUESTED)
    else:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only queued or running evaluations can be cancelled.",
        )

    await db.flush()
    await db.refresh(job)
    response = _create_job_response(job, paper)
    await db.commit()
    return response


async def _load_event_snapshot(job_id: str) -> dict | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EvaluationJob, Paper)
            .join(Paper, EvaluationJob.paper_id == Paper.id)
            .where(EvaluationJob.job_id == job_id)
        )
        row = result.one_or_none()
        if row is None:
            return None
        job, paper = row
        completed = TOTAL_DIMENSIONS if job.status == "completed" else max(
            0,
            min(job.completed_dimensions or 0, TOTAL_DIMENSIONS),
        )
        return {
            "job_id": job.job_id,
            "paper_id": paper.paper_id,
            "status": job.status,
            "state_version": job.state_version or 1,
            "completed_dimensions": completed,
            "total_dimensions": TOTAL_DIMENSIONS,
            "progress": round(completed / TOTAL_DIMENSIONS, 4),
            "current_dimension": job.current_dimension,
            "retry_count": job.retry_count or 0,
            "error_code": job.error_code,
            "error_message": sanitize_public_payload(job.error_message),
        }


@router.get("/{job_id}/events")
async def evaluation_events(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    exists = await db.scalar(
        select(EvaluationJob.id).where(EvaluationJob.job_id == job_id)
    )
    if exists is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation job not found.",
        )

    stream = evaluation_event_stream(
        request=request,
        job_id=job_id,
        last_event_id=request.headers.get("Last-Event-ID"),
        snapshot_loader=_load_event_snapshot,
        poll_seconds=settings.academic_sse_poll_seconds,
        heartbeat_seconds=settings.academic_sse_heartbeat_seconds,
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _completed_export_source(
    db: AsyncSession,
    job_id: str,
) -> tuple[EvaluationJob, Paper]:
    result = await db.execute(
        select(EvaluationJob, Paper)
        .join(Paper, EvaluationJob.paper_id == Paper.id)
        .where(EvaluationJob.job_id == job_id)
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation job not found.",
        )
    job, paper = row
    if job.status != "completed" or job.result_json is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Evaluation result is not available.",
        )
    return job, paper


@router.get("/{job_id}/export.json")
async def export_evaluation_json(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    job, paper = await _completed_export_source(db, job_id)
    export = build_result_export(job=job, paper=paper)
    return Response(
        content=json.dumps(export, ensure_ascii=False),
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{job.job_id}.academic-evaluation.v1.json"'
            )
        },
    )


@router.get("/{job_id}/export.pdf")
async def export_evaluation_pdf(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    job, paper = await _completed_export_source(db, job_id)
    export = build_result_export(job=job, paper=paper)
    return Response(
        content=render_result_pdf(export),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{job.job_id}.academic-evaluation.v1.pdf"'
            )
        },
    )


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_evaluation(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    try:
        deleted = await delete_evaluation_job(
            db,
            job_id=job_id,
            pdf_storage_root=settings.academic_pdf_storage_dir,
            checkpoint_storage_root=(
                settings.academic_checkpoint_storage_dir
            ),
        )
    except ActiveJobDeletionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Queued or running evaluations cannot be deleted.",
        ) from exc

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation job not found.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
