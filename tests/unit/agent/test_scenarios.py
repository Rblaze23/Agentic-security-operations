"""Replay of the four real scenarios recorded by scripts/record_agent_scenarios.py.

Every request is served from the fixtures by hash, so these tests cost nothing and fail loudly
(UnrecordedRequestError) when a prompt, tool schema or alert changes. The scenario-specific
expectations below state what the recorded run produced; see docs/agent.md for the analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from secops.agent.graph import AgentDeps, InvestigationResult, run_investigation
from secops.agent.llm import LLM
from secops.agent.replay import replay_registry
from secops.schemas.alert import Alert
from secops.tools.registry import ToolRegistry

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "llm"
SCENARIOS = ["ftp_bruteforce", "internal_portscan", "benign_high_score", "heartbleed"]
pytestmark = pytest.mark.skipif(
    not (FIXTURE_ROOT / "ftp_bruteforce" / "result.json").exists(),
    reason="scenario fixtures not recorded",
)


def _replay(name: str, registry: ToolRegistry) -> tuple[InvestigationResult, dict[str, Any]]:
    d = FIXTURE_ROOT / name
    alert = Alert.model_validate_json((d / "alert.json").read_text())
    recorded = json.loads((d / "result.json").read_text())
    deps = AgentDeps(
        registry=replay_registry(registry, d / "tools"),
        investigator=LLM("claude-opus-5-5", mode="replay", fixture_dir=d / "investigator"),
        critic=LLM("claude-sonnet-5-5", mode="replay", fixture_dir=d / "critic"),
        tool_budget=12,
    )
    return run_investigation(alert, deps, investigation_id=name), recorded


@pytest.mark.parametrize("name", SCENARIOS)
def test_replay_reproduces_the_recorded_run(name: str, full_registry: ToolRegistry) -> None:
    result, recorded = _replay(name, full_registry)
    r = result.report
    assert r.verdict == recorded["verdict"] and r.severity == recorded["severity"]
    assert [c.tool for c in result.tool_calls] == recorded["tool_calls"]
    assert result.iteration == recorded["critic_rejections"]
    assert result.usage.cost_usd == pytest.approx(recorded["cost_usd"])
    ids = {e.evidence_id for e in result.evidence}
    for f in r.findings:
        if f.kind == "observed":
            assert f.evidence_ids and set(f.evidence_ids) <= ids
    for t in r.attack_techniques:
        assert set(t.evidence_ids) <= ids
    for c in r.cves:
        assert set(c.evidence_ids) <= ids
    assert (
        r.investigation_steps
        and r.model_prediction.attack_probability > r.model_prediction.threshold
    )
