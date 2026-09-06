"""add public evaluation error code

Revision ID: c2a8d61f4e73
Revises: 8f7c2e1a6b90
Create Date: 2026-08-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c2a8d61f4e73"
down_revision: Union[str, Sequence[str], None] = "8f7c2e1a6b90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "academic_evaluation_jobs",
        sa.Column("error_code", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("academic_evaluation_jobs", "error_code")
