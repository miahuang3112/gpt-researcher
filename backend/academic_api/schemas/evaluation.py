from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


PUBLIC_API_SCHEMA_VERSION = "academic-evaluation.v1"


class VersionedResponse(BaseModel):
    schema_version: Literal["academic-evaluation.v1"] = (
        PUBLIC_API_SCHEMA_VERSION
    )


class EvaluationJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CreateEvaluationRequest(BaseModel):
    paper_id: str | None = Field(default=None, min_length=1, max_length=255)
    pdf_path: str = Field(min_length=1, max_length=1000)

    title: str | None = Field(default=None, max_length=500)
    source_url: str | None = Field(default=None, max_length=1000)
    model_name: str = Field(
        default="qwen3.5:9b",
        min_length=1,
        max_length=255,
    )
    max_chunks: int = Field(default=4, ge=1, le=20)


class CreateEvaluationResponse(VersionedResponse):
    job_id: str
    paper_id: str
    status: EvaluationJobStatus


class EvaluationJobResponse(VersionedResponse):
    job_id: str
    paper_id: str
    title: str | None = None
    status: EvaluationJobStatus
    state_version: int = Field(default=1, ge=1)
    retry_count: int = Field(default=0, ge=0)

    completed_dimensions: int = Field(
        default=0,
        ge=0,
        le=9,
    )

    total_dimensions: int = Field(
        default=9,
        ge=1,
    )

    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    current_dimension: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    cancelled_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )

    result: dict | None = None

    error_code: str | None = None
    error_message: str | None = None


class EvaluationJobSummary(VersionedResponse):
    job_id: str
    paper_id: str
    title: str | None = None
    status: EvaluationJobStatus
    state_version: int = Field(ge=1)
    retry_count: int = Field(ge=0)
    completed_dimensions: int = Field(ge=0, le=9)
    total_dimensions: int = 9
    progress: float = Field(ge=0.0, le=1.0)
    current_dimension: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class EvaluationJobListResponse(VersionedResponse):
    items: list[EvaluationJobSummary]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class EvaluationResultResponse(VersionedResponse):
    job_id: str
    paper_id: str
    status: EvaluationJobStatus
    completed_at: datetime
    result: dict
