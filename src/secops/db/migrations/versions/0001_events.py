"""events and load_runs tables

Revision ID: 0001_events
Revises:
Create Date: 2026-10-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_events"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("event_id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column("source_row_id", sa.BigInteger(), nullable=False),
        sa.Column("day", sa.String(10), nullable=False),
        sa.Column("ts_us", sa.BigInteger(), nullable=False),
        sa.Column("source_ip", sa.String(45), nullable=False),
        sa.Column("source_port", sa.Integer(), nullable=False),
        sa.Column("destination_ip", sa.String(45), nullable=False),
        sa.Column("destination_port", sa.Integer(), nullable=False),
        sa.Column("protocol", sa.SmallInteger(), nullable=False),
        sa.Column("flow_duration_us", sa.BigInteger(), nullable=False),
        sa.Column("fwd_packets", sa.Integer(), nullable=False),
        sa.Column("bwd_packets", sa.Integer(), nullable=False),
        sa.Column("fwd_bytes", sa.BigInteger(), nullable=False),
        sa.Column("bwd_bytes", sa.BigInteger(), nullable=False),
        sa.Column("syn_count", sa.Integer(), nullable=False),
        sa.Column("fin_count", sa.Integer(), nullable=False),
        sa.Column("rst_count", sa.Integer(), nullable=False),
        sa.Column("ack_count", sa.Integer(), nullable=False),
        sa.Column("feature_spec_version", sa.String(32), nullable=False),
        sa.Column("features", sa.LargeBinary(), nullable=False),
        sa.Column("label_raw", sa.String(64), nullable=False),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("family", sa.String(32), nullable=False),
        sa.Column("is_attack", sa.SmallInteger(), nullable=False),
        sa.Column("split_chrono", sa.String(5), nullable=False),
        sa.Column("split_heldout", sa.String(5), nullable=False),
    )
    op.create_index("ix_events_source_ip_ts", "events", ["source_ip", "ts_us"])
    op.create_index("ix_events_destination_ip_ts", "events", ["destination_ip", "ts_us"])
    op.create_index("ix_events_ts", "events", ["ts_us"])
    op.create_index("ix_events_destination_port", "events", ["destination_port"])
    op.create_table(
        "load_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("loaded_at", sa.DateTime(), nullable=False),
        sa.Column("source_path", sa.String(512), nullable=False),
        sa.Column("attempted_policy", sa.String(32), nullable=False),
        sa.Column("feature_spec_version", sa.String(32), nullable=False),
        sa.Column("manifest_digest", sa.String(64), nullable=False),
        sa.Column("git_sha", sa.String(40), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("seconds", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("load_runs")
    for name in (
        "ix_events_destination_port",
        "ix_events_ts",
        "ix_events_destination_ip_ts",
        "ix_events_source_ip_ts",
    ):
        op.drop_index(name, table_name="events")
    op.drop_table("events")
