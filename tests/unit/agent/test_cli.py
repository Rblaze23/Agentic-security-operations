"""secops-agent: alert from a stored event, investigate --alert-json end to end, show/recent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from typer.testing import CliRunner

from secops.agent import cli
from secops.agent.llm import LLM
from secops.agent.settings import AgentSettings
from secops.api.detector import DetectorService
from secops.db.events_loader import event_id_for
from secops.tools.registry import ToolRegistry
from tests.unit.agent.test_graph import _alert, _critic_ok, _draft_msg, _plan_msg, _tool_call
from tests.unit.agent.test_llm import FakeClient

runner = CliRunner()


def test_alert_from_event_scores_the_stored_flow(
    fixture_service: DetectorService, event_engine: Engine, flows: Any
) -> None:
    row = flows[flows["label"].astype(str) == "FTP-Patator"].iloc[0]
    eid = event_id_for(str(row["day"]), int(row["id"]))
    alert = cli.alert_from_event(eid, event_engine, fixture_service)
    assert alert.event_id == str(eid)
    assert alert.prediction.is_alert and alert.prediction.attack_probability > 0.5
    assert alert.metadata.destination_port == 21
    assert str(alert.metadata.source_ip) == str(row["Src IP"])
    assert alert.key_features  # top SHAP contributions become the alert's key features
    with pytest.raises(KeyError):
        cli.alert_from_event(999_999_999, event_engine, fixture_service)


def test_make_llms_requirements(tmp_path: Path) -> None:
    no_key = AgentSettings(ANTHROPIC_API_KEY=None)  # type: ignore[call-arg]
    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
        cli.make_llms("live", None, tmp_path, "medium", no_key)
    with pytest.raises(Exception, match="scenario"):
        cli.make_llms("replay", None, tmp_path, "medium", no_key)
    inv, crit = cli.make_llms("replay", "s1", tmp_path, "low", no_key)
    assert inv.model == "claude-opus-5-5" and crit.model == "claude-sonnet-5-5"
    assert inv.fixture_dir == tmp_path / "s1" / "investigator"
    assert crit.fixture_dir == tmp_path / "s1" / "critic"


def test_investigate_alert_json_then_show_and_recent(
    registry: ToolRegistry, flows: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alert = _alert(flows)
    alert_path = tmp_path / "alert.json"
    alert_path.write_text(alert.model_dump_json())
    msgs = [
        _plan_msg(),
        _tool_call(
            "t1", "get_related_events", {"event_id": int(alert.event_id), "window_minutes": 5}
        ),
        _draft_msg(findings=[{"kind": "observed", "statement": "burst", "evidence_ids": ["E1"]}]),
    ]

    def fake_llms(*_a: Any, **_k: Any) -> tuple[LLM, LLM]:
        return (
            LLM("claude-opus-5-5", client=FakeClient(msgs)),
            LLM("claude-sonnet-5-5", client=FakeClient([_critic_ok()])),
        )

    monkeypatch.setattr(cli, "make_llms", fake_llms)
    monkeypatch.setattr(cli, "_build", lambda *_a, **_k: (registry, None))
    url = f"sqlite:///{tmp_path / 'inv.db'}"
    out = tmp_path / "report.json"
    res = runner.invoke(
        cli.app,
        [
            "investigate",
            "--alert-json",
            str(alert_path),
            "--database-url",
            url,
            "--no-detector-tool",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    report = json.loads(out.read_text())
    assert report["alert_id"] == alert.alert_id and report["verdict"] == "true_positive"
    m = re.search(r"investigation_id: (\w+)", res.output)
    assert m and "persisted." in res.output and "cost $" in res.output
    inv_id = m.group(1)

    res = runner.invoke(cli.app, ["show", inv_id, "--database-url", url])
    assert res.exit_code == 0, res.output
    assert "get_related_events" in res.output and '"verdict": "true_positive"' in res.output

    res = runner.invoke(cli.app, ["recent", "--database-url", url])
    assert res.exit_code == 0 and inv_id in res.output

    res = runner.invoke(cli.app, ["show", "missing", "--database-url", url])
    assert res.exit_code == 1


def test_investigate_rejects_ambiguous_input(tmp_path: Path) -> None:
    res = runner.invoke(cli.app, ["investigate"])
    assert res.exit_code != 0
    res = runner.invoke(cli.app, ["investigate", "--event-id", "1", "--alert-json", str(tmp_path)])
    assert res.exit_code != 0


def test_llm_critic_setting_and_flag() -> None:
    # default follows the Phase 5 measurement (docs/evaluation.md): rules-only critic
    assert AgentSettings(ANTHROPIC_API_KEY=None).llm_critic is False  # type: ignore[call-arg]
    assert AgentSettings(ANTHROPIC_API_KEY=None, SECOPS_LLM_CRITIC="true").llm_critic is True  # type: ignore[call-arg]
