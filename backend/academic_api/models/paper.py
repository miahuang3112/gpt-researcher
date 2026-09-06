from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.academic_api.db.base import Base


class Paper(Base):
    __tablename__ = "academic_papers"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )
    paper_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    source_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
    )
    pdf_path: Mapped[str | None] = mapped_column(
        String(2048),
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

    evaluation_jobs: Mapped[list["EvaluationJob"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
    )