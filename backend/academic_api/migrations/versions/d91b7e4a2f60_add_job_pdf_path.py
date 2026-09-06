"""add job-owned managed PDF path

Revision ID: d91b7e4a2f60
Revises: c2a8d61f4e73
Create Date: 2026-08-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d91b7e4a2f60"
down_revision: Union[str, Sequence[str], None] = "c2a8d61f4e73"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "academic_evaluation_jobs",
        sa.Column("pdf_path", sa.String(length=2048), nullable=True),
    )
    op.execute(
        "UPDATE academic_evaluation_jobs AS jobs "
        "SET pdf_path = ("
        "SELECT papers.pdf_path FROM academic_papers AS papers "
        "WHERE papers.id = jobs.paper_id"
        ")"
    )


def downgrade() -> None:
    op.drop_column("academic_evaluation_jobs", "pdf_path")
