"""CI smoke evaluation: five golden cases replayed from recorded fixtures (model responses,
tool outputs and the alert itself), compared with the scores stored at record time. Zero cost,
no event store, no bundles, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import Engine

from secops.agent.settings import AgentSettings
from secops.evaluation.golden import load_golden_set
from secops.evaluation.runner import RunConfig, run_golden
from secops.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[3]
SMOKE_DIR = ROOT / "tests" / "fixtures" / "eval" / "smoke"
GOLDEN = ROOT / "evaluation" / "golden" / "v1.json"
EXPECTED = SMOKE_DIR / "expected.json"  # case_id -> {verdict, severity, grounding_rate, cost_usd}

pytestmark = pytest.mark.skipif(not EXPECTED.exists(), reason="smoke fixtures not recorded")


def test_smoke_cases_replay_to_their_recorded_scores(
    tmp_path: Path, event_engine: Engine, full_registry: ToolRegistry
) -> None:
    expected = json.loads(EXPECTED.read_text())
    case_ids = sorted(expected)
    assert {c.case_id for c in load_golden_set(GOLDEN).cases} >= set(case_ids)
    config = RunConfig(
        run_id="smoke",
        golden_path=GOLDEN,
        investigator="agent",
        mode="replay",
        recordings_dir=SMOKE_DIR,
        repeats=1,
        only=case_ids,
        judge=False,
    )
    record = run_golden(
        config,
        event_engine,
        full_registry,
        None,
        tmp_path / "smoke.json",
        settings=AgentSettings(ANTHROPIC_API_KEY=None),  # type: ignore[call-arg]
    )
    assert record.metrics is not None and record.metrics.cases == len(case_ids)
    for r in record.results:
        exp = expected[r.case_id]
        assert r.score.verdict == exp["verdict"], r.case_id
        assert r.score.severity == exp["severity"], r.case_id
        assert r.score.grounding_rate == pytest.approx(exp["grounding_rate"]), r.case_id
        assert r.score.cost_usd == pytest.approx(exp["cost_usd"]), r.case_id
        assert r.score.unsupported_refs == 0
