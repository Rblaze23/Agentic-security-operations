"""model_predictions table (API predictions persisted)

Revision ID: 0003_predictions
Revises: 0002_investigations
Create Date: 2026-10-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_predictions"
down_revision = "0002_investigations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_predictions",
        sa.Column("prediction_id", sa.String(32), primary_key=True),
        sa.Column("event_id", sa.String(128), nullable=True),
        sa.Column("alert_id", sa.String(64), nullable=True),
        sa.Column("attack_probability", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("is_alert", sa.Boolean(), nullable=False),
        sa.Column("predicted_family", sa.String(32), nullable=True),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_model_predictions_created_at", "model_predictions", ["created_at"])
    op.create_index("ix_model_predictions_event_id", "model_predictions", ["event_id"])


def downgrade() -> None:
    op.drop_index("ix_model_predictions_event_id", table_name="model_predictions")
    op.drop_index("ix_model_predictions_created_at", table_name="model_predictions")
    op.drop_table("model_predictions")
