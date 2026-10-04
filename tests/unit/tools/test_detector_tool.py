"""predict_attack: scores stored vectors through the same path as the API, leaks no labels."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from secops.api.detector import DetectorService
from secops.db.events_loader import event_id_for, unpack_features
from secops.db.models import Event
from secops.schemas.flow import PredictRequest
from secops.schemas.tools import PredictAttackInput, PredictAttackResult
from secops.tools.detector import DetectorTool, FeatureSpecMismatchError


@pytest.fixture(scope="module")
def tool(fixture_service: DetectorService, event_engine: Engine) -> DetectorTool:
    return DetectorTool(fixture_service, event_engine)


def _ids(flows: Any, label: str, n: int) -> list[int]:
    rows = flows[flows["label"].astype(str) == label].head(n)
    return [event_id_for(str(r["day"]), int(r["id"])) for _, r in rows.iterrows()]


def _api_path(service: DetectorService, engine: Engine, event_id: int) -> Any:
    with Session(engine) as conn:
        e = conn.scalars(select(Event).where(Event.event_id == event_id)).one()
        vec = unpack_features(e.features)
    feats = {
        n: (None if math.isnan(v) else float(v))
        for n, v in zip(service.spec.names, vec, strict=True)
    }
    return service.predict([PredictRequest(event_id=str(event_id), features=feats)])[0]


def test_by_event_ids_matches_the_api_path(
    tool: DetectorTool, fixture_service: DetectorService, event_engine: Engine, flows: Any
) -> None:
    ids = _ids(flows, "FTP-Patator", 3) + _ids(flows, "BENIGN", 3)
    out = tool.predict_attack(PredictAttackInput(event_ids=ids))
    assert isinstance(out, PredictAttackResult)
    assert [p.event_id for p in out.predictions] == ids and out.missing_event_ids == []
    assert out.feature_spec_version == fixture_service.spec.version
    for p in out.predictions:
        api = _api_path(fixture_service, event_engine, p.event_id or 0)
        assert p.attack_probability == pytest.approx(api.attack_probability, abs=1e-9)
        assert p.is_alert == api.is_alert and p.threshold == api.threshold
        assert p.predicted_family == api.predicted_family
        assert [c.feature for c in p.top_contributions] == [
            c.feature for c in api.top_contributions
        ]
    assert any(p.is_alert for p in out.predictions[:3]), "fixture brute-force rows should alert"


def test_missing_and_duplicate_ids(tool: DetectorTool, flows: Any) -> None:
    known = _ids(flows, "BENIGN", 1)[0]
    out = tool.predict_attack(PredictAttackInput(event_ids=[known, 999_999_999, known]))
    assert [p.event_id for p in out.predictions] == [known]
    assert out.missing_event_ids == [999_999_999]
    out = tool.predict_attack(PredictAttackInput(event_ids=[999_999_999]))
    assert out.predictions == [] and out.missing_event_ids == [999_999_999]


def test_by_features_matches_the_api_path(
    tool: DetectorTool, fixture_service: DetectorService, event_engine: Engine, flows: Any
) -> None:
    eid = _ids(flows, "FTP-Patator", 1)[0]
    with Session(event_engine) as conn:
        vec = unpack_features(
            conn.scalars(select(Event).where(Event.event_id == eid)).one().features
        )
    feats = {
        n: (None if math.isnan(v) else float(v))
        for n, v in zip(fixture_service.spec.names, vec, strict=True)
    }
    out = tool.predict_attack(PredictAttackInput(features=feats))
    api = fixture_service.predict([PredictRequest(features=feats)])[0]
    assert len(out.predictions) == 1 and out.predictions[0].event_id is None
    assert out.predictions[0].attack_probability == pytest.approx(api.attack_probability)
    with pytest.raises(ValueError, match="feature set mismatch"):
        tool.predict_attack(PredictAttackInput(features={"not a feature": 1.0}))


def test_input_bounds() -> None:
    with pytest.raises(ValidationError):
        PredictAttackInput()
    with pytest.raises(ValidationError):
        PredictAttackInput(event_ids=[1], features={"a": 1.0})
    with pytest.raises(ValidationError):
        PredictAttackInput(event_ids=list(range(101)))
    with pytest.raises(ValidationError):
        PredictAttackInput(event_ids=[1], extra=1)  # type: ignore[call-arg]


def test_spec_mismatch_is_an_error_not_a_guess(
    fixture_service: DetectorService,
    event_engine: Engine,
    flows: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = DetectorTool(fixture_service, event_engine)
    monkeypatch.setattr(fixture_service.spec, "version", "v9-other")
    with pytest.raises(FeatureSpecMismatchError):
        tool.predict_attack(PredictAttackInput(event_ids=_ids(flows, "BENIGN", 1)))


def test_output_schema_has_no_ground_truth(tool: DetectorTool) -> None:
    from tests.unit.tools.test_registry import GROUND_TRUTH, _property_names

    names = _property_names(tool.specs()[0].output_schema())
    assert not names & GROUND_TRUTH, names & GROUND_TRUTH
    assert "is_alert" in names and "attack_probability" in names
