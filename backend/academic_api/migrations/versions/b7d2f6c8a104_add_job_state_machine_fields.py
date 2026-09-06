"""add persisted evaluation job state machine fields

Revision ID: b7d2f6c8a104
Revises: a6c4e9f1b203
Create Date: 2026-08-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7d2f6c8a104"
down_revision: Union[str, Sequence[str], None] = "a6c4e9f1b203"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "academic_evaluation_jobs",
        sa.Column(
            "state_version",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
    )
    op.add_column(
        "academic_evaluation_jobs",
        sa.Column(
            "retry_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.add_column(
        "academic_evaluation_jobs",
        sa.Column("cancel_requested_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "academic_evaluation_jobs",
        sa.Column("cancelled_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("academic_evaluation_jobs", "cancelled_at")
    op.drop_column("academic_evaluation_jobs", "cancel_requested_at")
    op.drop_column("academic_evaluation_jobs", "retry_count")
    op.drop_column("academic_evaluation_jobs", "state_version")
