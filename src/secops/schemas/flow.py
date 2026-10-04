"""Inbound flow records: metadata (never features) plus the exact feature dictionary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress

from secops.detection.features import FeatureSpec

BATCH_LIMIT = 1000


class FeatureMismatchError(ValueError):
    """The request's feature names do not match the bundle's feature spec."""


class FlowMetadata(BaseModel):
    """Identifiers and context for a flow. Carried through to the alert, never used as features."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime | None = None
    source_ip: IPvAnyAddress | None = None
    destination_ip: IPvAnyAddress | None = None
    source_port: int | None = Field(default=None, ge=0, le=65535)
    destination_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: int | None = Field(default=None, ge=0, le=255)


def validate_features(features: Mapping[str, float | None], spec: FeatureSpec) -> dict[str, float]:
    """Return the features in spec order; null becomes NaN. Raises ValueError on any name drift."""
    names = set(spec.names)
    missing = [n for n in spec.names if n not in features]
    unknown = sorted(k for k in features if k not in names)
    if missing or unknown:
        raise FeatureMismatchError(
            f"feature set mismatch: missing={missing[:10]} unknown={unknown[:10]}"
        )
    out: dict[str, float] = {}
    for n in spec.names:
        v = features[n]
        out[n] = math.nan if v is None else float(v)
    return out


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    event_id: str | None = Field(default=None, max_length=128)
    metadata: FlowMetadata = Field(default_factory=FlowMetadata)
    features: dict[str, float | None] = Field(min_length=1)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictRequest] = Field(min_length=1, max_length=BATCH_LIMIT)
