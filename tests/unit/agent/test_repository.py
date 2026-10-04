"""Investigations round-trip through migration 0002 with their tool calls and cost."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from secops.agent.graph import run_investigation
from secops.agent.repository import InvestigationRepository
from secops.db.session import make_engine, upgrade_to_head
from secops.tools.registry import ToolRegistry
from tests.unit.agent.test_graph import (
    _alert,
    _critic_ok,
    _deps,
    _draft_msg,
    _plan_msg,
    _tool_call,
)


@pytest.fixture
def migrated_engine(tmp_path: Path) -> Engine:
    url = f"sqlite:///{tmp_path / 'inv.db'}"
    upgrade_to_head(url)
    return make_engine(url)


def _run(registry: ToolRegistry, flows: Any) -> Any:
    alert = _alert(flows)
    investigator = [
        _plan_msg(),
        _tool_call(
            "t1", "get_related_events", {"event_id": int(alert.event_id), "window_minutes": 5}
        ),
        _tool_call("t2", "get_asset", {"ip": str(alert.metadata.destination_ip)}),
        _draft_msg(findings=[{"kind": "observed", "statement": "burst", "evidence_ids": ["E1"]}]),
    ]
    return run_investigation(alert, _deps(registry, investigator, [_critic_ok()]))


def test_save_get_recent_round_trip(
    registry: ToolRegistry, flows: Any, migrated_engine: Engine
) -> None:
    result = _run(registry, flows)
    repo = InvestigationRepository(migrated_engine)
    when = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    assert repo.save(result, now=when) == result.investigation_id

    stored = repo.get(result.investigation_id)
    assert stored is not None
    assert stored.report == result.report
    assert stored.alert_id == result.report.alert_id
    assert stored.event_id == int(result.event_id)
    assert stored.created_at == when
    assert [c.tool for c in stored.tool_calls] == ["get_related_events", "get_asset"]
    assert stored.tool_calls[0].evidence_id == "E1" and stored.tool_calls[0].status == "ok"
    assert stored.cost_usd == result.usage.cost_usd and stored.cost_usd > 0
    assert stored.input_tokens == result.usage.input_tokens
    assert stored.model_investigator == "claude-opus-5-5"
    assert stored.model_critic == "claude-sonnet-5-5"
    assert stored.prompt_version == result.prompt_version

    rows = repo.recent()
    assert [r.investigation_id for r in rows] == [result.investigation_id]
    assert rows[0].verdict == result.report.verdict and rows[0].tool_call_count == 2
    assert repo.get("nope") is None
