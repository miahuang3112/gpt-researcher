"""create academic evaluation tables

Revision ID: 5b4d7fbb4d3b
Revises:
Create Date: 2026-08-25 21:22:07.565742

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5b4d7fbb4d3b"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "academic_papers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("paper_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("pdf_path", sa.String(length=2048), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_academic_papers_paper_id",
        "academic_papers",
        ["paper_id"],
        unique=True,
    )

    op.create_table(
        "academic_evaluation_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("paper_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("max_chunks", sa.Integer(), nullable=False),
        sa.Column("completed_dimensions", sa.Integer(), nullable=False),
        sa.Column("current_dimension", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("checkpoint_path", sa.String(length=2048), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["academic_papers.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_academic_evaluation_jobs_job_id",
        "academic_evaluation_jobs",
        ["job_id"],
        unique=True,
    )
    op.create_index(
        "ix_academic_evaluation_jobs_paper_id",
        "academic_evaluation_jobs",
        ["paper_id"],
        unique=False,
    )
    op.create_index(
        "ix_academic_evaluation_jobs_status",
        "academic_evaluation_jobs",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Let DROP TABLE remove indexes and the paper_id foreign key atomically.
    # MySQL rejects dropping an index that is still required by a foreign key.
    op.drop_table("academic_evaluation_jobs")
    op.drop_table("academic_papers")
