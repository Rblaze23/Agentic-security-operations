"""An alert: a prediction above the operating threshold, with its flow context."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from secops.schemas.flow import FlowMetadata, PredictRequest
from secops.schemas.prediction import Prediction


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_id: str | None = None
    metadata: FlowMetadata
    prediction: Prediction
    key_features: dict[str, float | None]
    status: Literal["new"] = "new"

    @classmethod
    def from_prediction(cls, request: PredictRequest, prediction: Prediction) -> Alert:
        return cls(
            event_id=request.event_id,
            metadata=request.metadata,
            prediction=prediction,
            key_features={c.feature: c.value for c in prediction.top_contributions},
        )
