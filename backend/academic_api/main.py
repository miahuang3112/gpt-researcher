import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from time import perf_counter

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.academic_api.cache.redis_client import close_redis_connection, redis_client
from backend.academic_api.db.session import get_db
from backend.academic_api.health import api_readiness_report
from backend.academic_api.metrics import (
    PROMETHEUS_CONTENT_TYPE,
    collect_academic_metrics,
    render_prometheus,
)
from backend.academic_api.observability import (
    elapsed_milliseconds,
    get_logger,
    log_event,
    normalize_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)

from backend.academic_api.routers.evaluations import (
    router as evaluations_router,
)
from backend.academic_api.services.outbox import run_outbox_dispatcher


logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    dispatcher_task = asyncio.create_task(
        run_outbox_dispatcher(),
        name="academic-outbox-dispatcher",
    )

    try:
        yield
    finally:
        dispatcher_task.cancel()
        with suppress(asyncio.CancelledError):
            await dispatcher_task
        await close_redis_connection()

app = FastAPI(
    title="Academic Evaluation API",
    lifespan=lifespan,
)

app.include_router(
    evaluations_router,
    prefix="/api",
    include_in_schema=False,
)
app.include_router(evaluations_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def public_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    log_event(
        logger,
        "http.unhandled_exception",
        level=logging.ERROR,
        method=request.method,
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.middleware("http")
async def request_observability(request: Request, call_next):
    correlation_id = normalize_correlation_id(
        request.headers.get("X-Request-ID")
    )
    token = set_correlation_id(correlation_id)
    started_at = perf_counter()
    log_event(
        logger,
        "http.request.started",
        method=request.method,
        path=request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        log_event(
            logger,
            "http.request.failed",
            level=logging.ERROR,
            method=request.method,
            path=request.url.path,
            duration_ms=elapsed_milliseconds(started_at),
            error_type=type(exc).__name__,
        )
        raise
    else:
        response.headers["X-Request-ID"] = correlation_id
        log_event(
            logger,
            "http.request.completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=elapsed_milliseconds(started_at),
        )
        return response
    finally:
        reset_correlation_id(token)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness() -> JSONResponse:
    report = await api_readiness_report()
    response_status = (
        status.HTTP_200_OK
        if report["status"] == "ready"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(
        content=report,
        status_code=response_status,
    )


@app.get("/metrics", include_in_schema=False)
async def metrics(
    db: AsyncSession = Depends(get_db),
) -> Response:
    snapshot = await collect_academic_metrics(db, redis_client)
    return Response(
        content=render_prometheus(snapshot),
        media_type=PROMETHEUS_CONTENT_TYPE,
    )
