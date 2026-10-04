"""ORM models. `Event` holds one flow with its metadata, hidden ground truth and feature vector."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Event(Base):
    """One network flow. Timestamps are stored as UTC microseconds since the epoch (`ts_us`) so
    range queries are plain integer comparisons on every backend. `event_id` is
    `day_index * 1_000_000 + source_row_id` because the dataset's `id` restarts at 1 per day."""

    __tablename__ = "events"

    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    source_row_id: Mapped[int] = mapped_column(BigInteger, nullable=False)  # dataset `id` (per day)
    day: Mapped[str] = mapped_column(String(10), nullable=False)
    ts_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_ip: Mapped[str] = mapped_column(String(45), nullable=False)
    source_port: Mapped[int] = mapped_column(Integer, nullable=False)
    destination_ip: Mapped[str] = mapped_column(String(45), nullable=False)
    destination_port: Mapped[int] = mapped_column(Integer, nullable=False)
    protocol: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    flow_duration_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fwd_packets: Mapped[int] = mapped_column(Integer, nullable=False)
    bwd_packets: Mapped[int] = mapped_column(Integer, nullable=False)
    fwd_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bwd_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    syn_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fin_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rst_count: Mapped[int] = mapped_column(Integer, nullable=False)
    ack_count: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_spec_version: Mapped[str] = mapped_column(String(32), nullable=False)
    features: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Ground truth: used only to build evaluation sets. No tool returns these columns.
    label_raw: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    family: Mapped[str] = mapped_column(String(32), nullable=False)
    is_attack: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    split_chrono: Mapped[str] = mapped_column(String(5), nullable=False)
    split_heldout: Mapped[str] = mapped_column(String(5), nullable=False)

    __table_args__ = (
        Index("ix_events_source_ip_ts", "source_ip", "ts_us"),
        Index("ix_events_destination_ip_ts", "destination_ip", "ts_us"),
        Index("ix_events_ts", "ts_us"),
        Index("ix_events_destination_port", "destination_port"),
    )


class LoadRun(Base):
    """Provenance of each `load-events` run."""

    __tablename__ = "load_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    loaded_at: Mapped[datetime] = mapped_column(nullable=False)
    source_path: Mapped[str] = mapped_column(String(512), nullable=False)
    attempted_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_spec_version: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    git_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    rows: Mapped[int] = mapped_column(Integer, nullable=False)
    seconds: Mapped[float] = mapped_column(nullable=False)


class Investigation(Base):
    """One agent investigation of one alert: outcome, cost and the full report as JSON."""

    __tablename__ = "investigations"

    investigation_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    alert_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    attack_family: Mapped[str | None] = mapped_column(String(32), nullable=True)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_call_count: Mapped[int] = mapped_column(Integer, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    model_investigator: Mapped[str] = mapped_column(String(64), nullable=False)
    model_critic: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    report_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="investigation", cascade="all, delete-orphan", order_by="ToolCall.id"
    )

    __table_args__ = (
        Index("ix_investigations_alert_id", "alert_id"),
        Index("ix_investigations_created_at", "created_at"),
    )


class ToolCall(Base):
    """One tool invocation inside an investigation, keyed by the evidence id it produced."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    investigation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("investigations.investigation_id"), nullable=False
    )
    evidence_id: Mapped[str] = mapped_column(String(16), nullable=False)
    tool: Mapped[str] = mapped_column(String(64), nullable=False)
    arguments_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    called_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    investigation: Mapped[Investigation] = relationship(back_populates="tool_calls")

    __table_args__ = (Index("ix_tool_calls_investigation_id", "investigation_id"),)


class ModelPrediction(Base):
    """One detector prediction. Schema only (migration 0003): nothing writes to it yet; the
    drift-monitoring future work is what would fill it."""

    __tablename__ = "model_predictions"

    prediction_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    alert_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attack_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    is_alert: Mapped[bool] = mapped_column(Boolean, nullable=False)
    predicted_family: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_model_predictions_created_at", "created_at"),
        Index("ix_model_predictions_event_id", "event_id"),
    )
