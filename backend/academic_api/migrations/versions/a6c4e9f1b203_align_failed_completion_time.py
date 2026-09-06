"""align failed job completion timestamps

Revision ID: a6c4e9f1b203
Revises: f3a1c8b7d902
Create Date: 2026-08-26
"""

from typing import Sequence, Union

from alembic import op


revision: str = "a6c4e9f1b203"
down_revision: Union[str, Sequence[str], None] = "f3a1c8b7d902"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE academic_evaluation_jobs "
        "SET completed_at = started_at "
        "WHERE status = 'failed' "
        "AND started_at IS NOT NULL "
        "AND (completed_at IS NULL OR completed_at < started_at)"
    )


def downgrade() -> None:
    # The inconsistent historical timestamp must not be restored.
    pass
