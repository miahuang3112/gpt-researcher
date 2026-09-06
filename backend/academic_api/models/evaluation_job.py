from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.academic_api.db.base import Base


class EvaluationJob(Base):
    __tablename__ = "academic_evaluation_jobs"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        index=True,
        nullable=False,
    )
    paper_id: Mapped[int] = mapped_column(
        ForeignKey(
            "academic_papers.id",
            ondelete="CASCADE",
        ),
        index=True,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        index=True,
        nullable=False,
    )
    state_version: Mapped[int] = mapped_column(
        default=1,
        server_default=text("1"),
        nullable=False,
    )
    retry_count: Mapped[int] = mapped_column(
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(
        String(255),
        default="qwen3.5:9b",
        nullable=False,
    )
    max_chunks: Mapped[int] = mapped_column(
        default=4,
        nullable=False,
    )
    completed_dimensions: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )
    current_dimension: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    checkpoint_path: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
    )
    pdf_path: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
    )
    result_json: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    paper: Mapped["Paper"] = relationship(
        back_populates="evaluation_jobs",
    )
    outbox_events: Mapped[list["OutboxEvent"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
