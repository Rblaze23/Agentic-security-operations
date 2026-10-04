"""Opt-in live check (`uv run pytest tests/llm -m llm`): one real investigation of the FTP
brute-force alert. Costs money; asserts only invariants, never exact model wording."""

from __future__ import annotations

from pathlib import Path

import pytest

from secops.agent.cli import _build, alert_from_event, make_llms
from secops.agent.graph import AgentDeps, run_investigation
from secops.agent.settings import get_agent_settings
from secops.config import get_settings
from secops.db.session import make_engine

pytestmark = pytest.mark.llm

FTP_EVENT = 1110604


def test_live_ftp_bruteforce_is_grounded() -> None:
    settings = get_settings()
    engine = make_engine(settings.resolved_database_url(), read_only=True)
    registry, service = _build(settings, engine, with_detector=True)
    assert service is not None
    alert = alert_from_event(FTP_EVENT, engine, service)
    investigator, critic = make_llms("live", None, Path("."), "medium", get_agent_settings())
    result = run_investigation(alert, AgentDeps(registry, investigator, critic, tool_budget=12))
    report = result.report
    ids = {e.evidence_id for e in result.evidence}
    for f in report.findings:
        if f.kind == "observed":
            assert f.evidence_ids and set(f.evidence_ids) <= ids
    assert report.verdict in ("true_positive", "needs_human_review")
    assert result.usage.cost_usd > 0 and result.tool_calls
