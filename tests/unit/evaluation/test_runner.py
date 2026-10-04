from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from secops.agent.settings import AgentSettings
from secops.api.detector import DetectorService
from secops.evaluation.golden import build_golden_set
from secops.evaluation.runner import RunConfig, RunRecord, estimated_cost_usd, run_golden
from secops.tools.registry import ToolRegistry

SETTINGS = AgentSettings(ANTHROPIC_API_KEY=None)  # type: ignore[call-arg]


@pytest.fixture
def golden_path(tmp_path: Path, event_engine: Engine, fixture_service: DetectorService) -> Path:
    gs = build_golden_set(
        event_engine, fixture_service, seed=3, per_family=1, benign=1, adversarial=1
    )
    p = tmp_path / "golden.json"
    p.write_text(gs.model_dump_json())
    return p


def _config(golden_path: Path, tmp_path: Path, **over: Any) -> RunConfig:
    base: dict[str, Any] = dict(
        run_id="t",
        golden_path=golden_path,
        investigator="baseline",
        mode="record",
        recordings_dir=tmp_path / "rec",
        repeats=2,
    )
    base.update(over)
    return RunConfig(**base)


def _n_cases(golden_path: Path) -> int:
    return len(json.loads(golden_path.read_text())["cases"])


def test_baseline_run_scores_every_case_twice_and_writes_incrementally(
    golden_path: Path,
    tmp_path: Path,
    event_engine: Engine,
    full_registry: ToolRegistry,
    fixture_service: DetectorService,
) -> None:
    out = tmp_path / "run.json"
    cfg = _config(golden_path, tmp_path)
    rec = run_golden(cfg, event_engine, full_registry, fixture_service, out, settings=SETTINGS)
    assert len(rec.results) == _n_cases(golden_path) * 2
    assert rec.metrics is not None and rec.metrics.repeats == 2
    assert rec.metrics.cost_total_usd == 0.0 and rec.models["investigator"] == "rule-based"
    assert rec.prompt_version == "baseline-v1" and rec.golden_version == "v1"
    saved = RunRecord.model_validate_json(out.read_text())
    assert len(saved.results) == len(rec.results) and saved.finished_at is not None
    assert rec.metrics.flaky_cases == []  # the baseline is deterministic
    assert estimated_cost_usd(cfg, 10) == 0.0
    assert estimated_cost_usd(
        _config(golden_path, tmp_path, investigator="agent", mode="record", repeats=1), 10
    ) == pytest.approx(2.0)


def test_runner_fails_loudly_on_missing_event(
    golden_path: Path,
    tmp_path: Path,
    event_engine: Engine,
    full_registry: ToolRegistry,
    fixture_service: DetectorService,
) -> None:
    gs = json.loads(golden_path.read_text())
    gs["cases"][0]["event_id"] = 999_999_999
    golden_path.write_text(json.dumps(gs, default=str))
    with pytest.raises(KeyError, match=gs["cases"][0]["case_id"]):
        run_golden(
            _config(golden_path, tmp_path, repeats=1),
            event_engine,
            full_registry,
            fixture_service,
            tmp_path / "r.json",
            settings=SETTINGS,
        )


def test_runner_resumes_completed_cases(
    golden_path: Path,
    tmp_path: Path,
    event_engine: Engine,
    full_registry: ToolRegistry,
    fixture_service: DetectorService,
) -> None:
    out = tmp_path / "run.json"
    first_id = json.loads(golden_path.read_text())["cases"][0]["case_id"]
    first = run_golden(
        _config(golden_path, tmp_path, repeats=1, only=[first_id]),
        event_engine,
        full_registry,
        fixture_service,
        out,
        settings=SETTINGS,
    )
    assert len(first.results) == 1
    second = run_golden(
        _config(golden_path, tmp_path, repeats=1),
        event_engine,
        full_registry,
        fixture_service,
        out,
        resume=True,
        settings=SETTINGS,
    )
    assert len(second.results) == _n_cases(golden_path)
    assert second.results[0] == first.results[0]  # not re-run
    with pytest.raises(KeyError, match="unknown golden case ids"):
        run_golden(
            _config(golden_path, tmp_path, repeats=1, only=["nope"]),
            event_engine,
            full_registry,
            fixture_service,
            tmp_path / "x.json",
            settings=SETTINGS,
        )


def test_agent_run_in_replay_mode_needs_recorded_fixtures(
    golden_path: Path,
    tmp_path: Path,
    event_engine: Engine,
    full_registry: ToolRegistry,
    fixture_service: DetectorService,
) -> None:
    from secops.agent.llm import UnrecordedRequestError

    cfg = _config(golden_path, tmp_path, investigator="agent", mode="replay", repeats=1)
    with pytest.raises(UnrecordedRequestError):
        run_golden(
            cfg,
            event_engine,
            full_registry,
            fixture_service,
            tmp_path / "a.json",
            settings=SETTINGS,
        )


def test_adversarial_cases_get_injected_tools_and_record_mode_saves_alerts(
    golden_path: Path,
    tmp_path: Path,
    event_engine: Engine,
    full_registry: ToolRegistry,
    fixture_service: DetectorService,
) -> None:
    gs = json.loads(golden_path.read_text())
    adv = next(c for c in gs["cases"] if c["kind"] == "adversarial")
    cfg = _config(golden_path, tmp_path, mode="record", repeats=1, only=[adv["case_id"]])
    rec = run_golden(
        cfg, event_engine, full_registry, fixture_service, tmp_path / "r.json", settings=SETTINGS
    )
    case_dir = tmp_path / "rec" / adv["case_id"] / "r0"
    assert (case_dir / "alert.json").exists()
    tools = sorted(p.name for p in (case_dir / "tools").iterdir())
    assert tools and all(n.endswith(".json.gz") for n in tools)
    assert rec.results[0].score.adversarial_resisted is not None
