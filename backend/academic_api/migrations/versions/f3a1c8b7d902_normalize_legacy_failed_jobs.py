"""normalize legacy failed evaluation jobs

Revision ID: f3a1c8b7d902
Revises: d91b7e4a2f60
Create Date: 2026-08-26
"""

from typing import Sequence, Union

from alembic import op


revision: str = "f3a1c8b7d902"
down_revision: Union[str, Sequence[str], None] = "d91b7e4a2f60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Failed rows created before public error codes may contain exception text
    # and no terminal timestamp. Normalize them without changing successful or
    # active jobs.
    op.execute(
        "UPDATE academic_evaluation_jobs "
        "SET error_code = 'evaluation_failed', "
        "error_message = 'The academic evaluation failed.', "
        "completed_at = COALESCE(completed_at, updated_at, created_at), "
        "current_dimension = NULL "
        "WHERE status = 'failed' AND error_code IS NULL"
    )


def downgrade() -> None:
    # Redacted internal exception text must not be restored.
    pass
