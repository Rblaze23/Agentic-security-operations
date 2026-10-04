from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import Engine

from secops.api.detector import DetectorService
from secops.evaluation.golden import (
    EXPECTED_BY_FAMILY,
    build_golden_set,
    expected_severity,
    load_golden_set,
)


def test_family_expectations_cover_every_attack_family() -> None:
    from secops.data.schema import FAMILIES

    assert set(EXPECTED_BY_FAMILY) == set(FAMILIES) - {"benign"}
    for fam, exp in EXPECTED_BY_FAMILY.items():
        assert exp["techniques"] and exp["tools"] and exp["evidence"], fam


def test_build_golden_set_is_deterministic_and_balanced(
    event_engine: Engine, fixture_service: DetectorService
) -> None:
    a = build_golden_set(
        event_engine, fixture_service, seed=7, per_family=2, benign=2, adversarial=1
    )
    b = build_golden_set(
        event_engine, fixture_service, seed=7, per_family=2, benign=2, adversarial=1
    )
    assert [c.case_id for c in a.cases] == [c.case_id for c in b.cases]
    assert {c.kind for c in a.cases} == {"attack", "benign_fp", "adversarial"}
    attacks = [c for c in a.cases if c.kind == "attack"]
    assert all(c.expected_verdict == "true_positive" for c in attacks)
    assert all(c.expected_family in EXPECTED_BY_FAMILY for c in attacks)
    benign = [c for c in a.cases if c.kind == "benign_fp"]
    assert benign and all(
        c.expected_verdict == "false_positive" and c.detector_probability > 0 for c in benign
    )
    adv = [c for c in a.cases if c.kind == "adversarial"]
    assert adv and adv[0].injection and adv[0].case_id.endswith("-adv")
    assert adv[0].expected_verdict == "true_positive"
    assert len({c.case_id for c in a.cases}) == len(a.cases)


def test_expected_severity_uses_the_rubric() -> None:
    assert expected_severity("brute_force", "192.168.10.50", "true_positive") == "high"
    assert expected_severity("port_scan", "192.168.10.17", "true_positive") == "low"
    assert expected_severity("brute_force", None, "false_positive") == "low"


def test_round_trip_json(
    tmp_path: Path, event_engine: Engine, fixture_service: DetectorService
) -> None:
    gs = build_golden_set(
        event_engine, fixture_service, seed=1, per_family=1, benign=1, adversarial=1
    )
    path = tmp_path / "v1.json"
    path.write_text(gs.model_dump_json(indent=2))
    assert load_golden_set(path) == gs
    raw: dict[str, Any] = json.loads(path.read_text())
    assert "label" in raw["cases"][0]
