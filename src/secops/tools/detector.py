"""`predict_attack`: the Phase 2 detector as a read-only tool.

By `event_ids` the stored feature vectors are scored (so the agent can ask "do the neighbouring
flows also look like attacks?"); by `features` one hand-supplied dictionary goes through the same
validation path as the API. The tool returns scores, never the stored ground truth."""

from __future__ import annotations

import math
from ipaddress import ip_address
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from secops.api.detector import DetectorService
from secops.api.settings import ApiSettings, get_api_settings
from secops.config import Settings
from secops.db.events_loader import unpack_features
from secops.db.models import Event
from secops.detection.features import FeatureSpec
from secops.schemas.flow import FlowMetadata, PredictRequest
from secops.schemas.tools import (
    ContributionOut,
    PredictAttackInput,
    PredictAttackResult,
    ToolPrediction,
)
from secops.tools.base import ToolSpec
from secops.tools.events import _from_us


class FeatureSpecMismatchError(RuntimeError):
    """The event store was loaded with a different feature spec than the bundle expects."""


def request_for_event(e: Event, spec: FeatureSpec) -> PredictRequest:
    """The stored flow as the API would receive it: metadata apart, features in spec order."""
    if e.feature_spec_version != spec.version:
        raise FeatureSpecMismatchError(
            f"event {e.event_id} stored with feature spec {e.feature_spec_version!r}, "
            f"bundle expects {spec.version!r}"
        )
    vector = unpack_features(e.features)
    if len(vector) != len(spec.names):
        raise FeatureSpecMismatchError(
            f"event {e.event_id}: {len(vector)} stored features, spec has {len(spec.names)}"
        )
    feats = {
        n: (None if math.isnan(v) else float(v)) for n, v in zip(spec.names, vector, strict=True)
    }
    return PredictRequest(
        event_id=str(e.event_id),
        metadata=FlowMetadata(
            timestamp=_from_us(e.ts_us),
            source_ip=ip_address(e.source_ip),
            destination_ip=ip_address(e.destination_ip),
            source_port=e.source_port,
            destination_port=e.destination_port,
            protocol=e.protocol,
        ),
        features=feats,
    )


class DetectorTool:
    def __init__(self, service: DetectorService, engine: Engine) -> None:
        self.service = service
        self.engine = engine

    def _request_for(self, e: Event) -> PredictRequest:
        return request_for_event(e, self.service.spec)

    def predict_attack(self, inp: PredictAttackInput) -> PredictAttackResult:
        missing: list[int] = []
        if inp.event_ids is not None:
            wanted = list(dict.fromkeys(inp.event_ids))
            with Session(self.engine) as conn:
                rows = conn.scalars(select(Event).where(Event.event_id.in_(wanted))).all()
                by_id = {e.event_id: e for e in rows}
                requests = [self._request_for(by_id[i]) for i in wanted if i in by_id]
            missing = [i for i in wanted if i not in by_id]
        else:
            assert inp.features is not None
            requests = [PredictRequest(features=inp.features)]
        preds = self.service.predict(requests) if requests else []
        m = self.service.detector.manifest
        return PredictAttackResult(
            predictions=[
                ToolPrediction(
                    event_id=int(p.event_id) if p.event_id is not None else None,
                    attack_probability=p.attack_probability,
                    threshold=p.threshold,
                    is_alert=p.is_alert,
                    predicted_family=p.predicted_family,
                    family_probabilities=p.family_probabilities,
                    top_contributions=[
                        ContributionOut(feature=c.feature, value=c.value, shap_value=c.shap_value)
                        for c in p.top_contributions
                    ],
                )
                for p in preds
            ],
            missing_event_ids=missing,
            model_name=m.model_name,
            model_version=m.version,
            feature_spec_version=self.service.spec.version,
        )

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="predict_attack",
                description=(
                    "Score flows with the production intrusion detector: attack probability, the "
                    "operating threshold, whether each flow would alert, the predicted attack "
                    "family for alerts, and the top SHAP feature contributions. Pass up to 100 "
                    "event_ids from the event store (unknown ids are listed in missing_event_ids) "
                    "or exactly one features dictionary with the full feature set. It reads stored "
                    "feature vectors only and cannot reveal labels or change the model."
                ),
                input_model=PredictAttackInput,
                output_model=PredictAttackResult,
                run=self.predict_attack,
                external_source=None,
            )
        ]


def detector_specs(
    settings: Settings, engine: Engine, api_settings: ApiSettings | None = None
) -> list[ToolSpec]:
    """Load the exported bundles named by the API settings (SECOPS_MODEL_DIR) and wrap them.
    `settings` is accepted for signature parity with the other factories; bundle paths live in
    ApiSettings because the API and the tool must score with the same bundles."""
    api = api_settings or get_api_settings()
    family: Path | None = api.family_dir if api.family_dir.exists() else None
    service = DetectorService.from_dirs(api.detector_dir, family, k=api.top_k)
    return DetectorTool(service, engine).specs()
