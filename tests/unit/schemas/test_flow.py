import math

import pytest
from pydantic import ValidationError

from secops.data.schema import FEATURE_COLS
from secops.detection.features import FEATURE_SPEC_V1
from secops.schemas.flow import (
    BATCH_LIMIT,
    BatchPredictRequest,
    FlowMetadata,
    PredictRequest,
    validate_features,
)


def _features(**overrides: float | None) -> dict[str, float | None]:
    f: dict[str, float | None] = {n: 1.0 for n in FEATURE_COLS}
    f.update(overrides)
    return f


def test_flow_features_rejects_missing_and_unknown_names() -> None:
    bad = _features()
    del bad["Flow Duration"]
    bad["flow duration"] = 1.0  # wrong case counts as unknown
    with pytest.raises(ValueError) as exc:
        validate_features(bad, FEATURE_SPEC_V1)
    msg = str(exc.value)
    assert "missing=['Flow Duration']" in msg and "unknown=['flow duration']" in msg


def test_flow_features_null_becomes_nan() -> None:
    out = validate_features(_features(**{"Flow Bytes/s": None}), FEATURE_SPEC_V1)
    assert math.isnan(out["Flow Bytes/s"])
    assert list(out) == FEATURE_SPEC_V1.names  # spec order, always


def test_predict_request_forbids_extra_fields_and_non_finite() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(features=_features(), surprise=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        PredictRequest(features=_features(**{"Flow Duration": float("inf")}))
    with pytest.raises(ValidationError):
        PredictRequest(features=_features(**{"Flow Duration": float("nan")}))


def test_metadata_validates_ips_and_ports() -> None:
    m = FlowMetadata(source_ip="192.168.10.50", destination_port=80, protocol=6)
    assert str(m.source_ip) == "192.168.10.50"
    with pytest.raises(ValidationError):
        FlowMetadata(source_ip="not-an-ip")
    with pytest.raises(ValidationError):
        FlowMetadata(destination_port=70000)
    with pytest.raises(ValidationError):
        FlowMetadata(protocol=300)


def test_batch_bounds() -> None:
    one = PredictRequest(features=_features())
    assert len(BatchPredictRequest(items=[one]).items) == 1
    with pytest.raises(ValidationError):
        BatchPredictRequest(items=[])
    with pytest.raises(ValidationError):
        BatchPredictRequest(items=[one] * (BATCH_LIMIT + 1))


def test_event_id_length_capped() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(event_id="x" * 129, features=_features())
