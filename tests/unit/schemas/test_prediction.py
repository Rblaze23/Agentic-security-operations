import math
from datetime import UTC

import pytest
from pydantic import ValidationError

from secops.data.schema import FEATURE_COLS
from secops.schemas.alert import Alert
from secops.schemas.flow import FlowMetadata, PredictRequest
from secops.schemas.prediction import FeatureContribution, Prediction


def _prediction(**overrides: object) -> Prediction:
    base: dict[str, object] = {
        "event_id": "evt-1",
        "model_name": "secops-detector",
        "model_version": 1,
        "run_id": "abc",
        "feature_spec_version": "v1-noport",
        "attack_probability": 0.9,
        "threshold": 0.0002,
        "is_alert": True,
        "predicted_family": "dos",
        "family_probabilities": {"dos": 0.99, "ddos": 0.01},
        "top_contributions": [
            FeatureContribution(feature="Bwd Packet Length Std", value=12.5, shap_value=2.1),
            FeatureContribution(feature="Flow Bytes/s", value=None, shap_value=-0.3),
        ],
        "latency_ms": 3.2,
    }
    base.update(overrides)
    return Prediction(**base)  # type: ignore[arg-type]


def test_prediction_defaults_ids_and_utc_timestamp() -> None:
    p = _prediction()
    assert len(p.prediction_id) == 32
    assert p.timestamp.tzinfo is UTC
    assert _prediction().prediction_id != p.prediction_id


def test_prediction_bounds() -> None:
    with pytest.raises(ValidationError):
        _prediction(attack_probability=1.5)
    with pytest.raises(ValidationError):
        _prediction(latency_ms=-1)


def test_contribution_value_none_serialises_as_null_not_nan() -> None:
    p = _prediction()
    dumped = p.model_dump_json()
    assert "NaN" not in dumped and '"value":null' in dumped
    assert math.isnan(float("nan"))  # sanity: NaN is what None stands in for


def test_alert_from_prediction_picks_key_features() -> None:
    req = PredictRequest(
        event_id="evt-1",
        metadata=FlowMetadata(source_ip="172.16.0.1", destination_ip="192.168.10.50"),
        features={n: 1.0 for n in FEATURE_COLS},
    )
    p = _prediction()
    a = Alert.from_prediction(req, p)
    assert a.status == "new" and a.event_id == "evt-1" and len(a.alert_id) == 32
    assert a.key_features == {"Bwd Packet Length Std": 12.5, "Flow Bytes/s": None}
    assert a.prediction is p and str(a.metadata.destination_ip) == "192.168.10.50"
