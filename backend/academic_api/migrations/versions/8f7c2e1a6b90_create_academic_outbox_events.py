"""create academic outbox events

Revision ID: 8f7c2e1a6b90
Revises: 5b4d7fbb4d3b
Create Date: 2026-08-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8f7c2e1a6b90"
down_revision: Union[str, Sequence[str], None] = "5b4d7fbb4d3b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "academic_outbox_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["academic_evaluation_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_academic_outbox_events_event_id",
        "academic_outbox_events",
        ["event_id"],
        unique=True,
    )
    op.create_index(
        "ix_academic_outbox_events_job_id",
        "academic_outbox_events",
        ["job_id"],
        unique=False,
    )
    op.create_index(
        "ix_academic_outbox_pending",
        "academic_outbox_events",
        ["published_at", "next_attempt_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    # Dropping the table also drops its indexes and foreign key. In MySQL the
    # job_id index cannot be removed first because it supports that foreign key.
    op.drop_table("academic_outbox_events")
