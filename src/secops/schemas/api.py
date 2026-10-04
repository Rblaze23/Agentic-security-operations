"""HTTP response envelopes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from secops.schemas.alert import Alert
from secops.schemas.prediction import Prediction


class PredictResponse(BaseModel):
    prediction: Prediction
    alert: Alert | None = None


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    alerts: list[Alert]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"]
    detector_loaded: bool
    family_loaded: bool
    bundle_versions: dict[str, int] = Field(default_factory=dict)


class BundleInfo(BaseModel):
    model_name: str
    version: int
    run_id: str
    alias: str
    exported_at: datetime
    feature_spec_version: str
    model_kind: str
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    detector: BundleInfo
    family: BundleInfo | None
    feature_spec_version: str
    threshold: float
    top_k: int
