"""investigations and tool_calls tables

Revision ID: 0002_investigations
Revises: 0001_events
Create Date: 2026-10-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_investigations"
down_revision = "0001_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "investigations",
        sa.Column("investigation_id", sa.String(32), primary_key=True),
        sa.Column("alert_id", sa.String(64), nullable=False),
        sa.Column("event_id", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("verdict", sa.String(32), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("attack_family", sa.String(32), nullable=True),
        sa.Column("iterations", sa.Integer(), nullable=False),
        sa.Column("tool_call_count", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("model_investigator", sa.String(64), nullable=False),
        sa.Column("model_critic", sa.String(64), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.Column("report_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_investigations_alert_id", "investigations", ["alert_id"])
    op.create_index("ix_investigations_created_at", "investigations", ["created_at"])
    op.create_table(
        "tool_calls",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "investigation_id",
            sa.String(32),
            sa.ForeignKey("investigations.investigation_id"),
            nullable=False,
        ),
        sa.Column("evidence_id", sa.String(16), nullable=False),
        sa.Column("tool", sa.String(64), nullable=False),
        sa.Column("arguments_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("called_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_tool_calls_investigation_id", "tool_calls", ["investigation_id"])


def downgrade() -> None:
    op.drop_index("ix_tool_calls_investigation_id", table_name="tool_calls")
    op.drop_table("tool_calls")
    op.drop_index("ix_investigations_created_at", table_name="investigations")
    op.drop_index("ix_investigations_alert_id", table_name="investigations")
    op.drop_table("investigations")
