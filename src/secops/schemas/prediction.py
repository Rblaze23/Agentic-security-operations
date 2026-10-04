"""Model output for one flow."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class FeatureContribution(BaseModel):
    """One SHAP contribution. `value` is None when the feature was NaN (JSON has no NaN)."""

    feature: str
    value: float | None
    shap_value: float


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(default_factory=_new_id)
    event_id: str | None = None
    model_name: str
    model_version: int
    run_id: str
    feature_spec_version: str
    attack_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0)
    is_alert: bool
    predicted_family: str | None = None
    family_probabilities: dict[str, float] | None = None
    top_contributions: list[FeatureContribution] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)
    latency_ms: float = Field(ge=0.0)
