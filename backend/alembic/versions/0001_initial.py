"""initial schema: batches and predictions

Revision ID: 0001
Revises:
Create Date: 2026-10-04

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "batches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "batch_id",
            sa.Integer(),
            sa.ForeignKey(
                "batches.id", ondelete="CASCADE", name="fk_predictions_batch_id"
            ),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model", sa.String(length=50), nullable=False),
        sa.Column("source_date", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "label IN ('positive', 'neutral', 'negative')", name="ck_predictions_label"
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="ck_predictions_confidence"
        ),
    )
    op.create_index("ix_predictions_batch_id", "predictions", ["batch_id"])


def downgrade() -> None:
    op.drop_index("ix_predictions_batch_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("batches")
