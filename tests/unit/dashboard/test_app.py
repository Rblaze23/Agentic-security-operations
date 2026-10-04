"""The dashboard against a fake client: empty state, a replayed report, evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "dashboard" / "app.py"
REPORT = json.loads(
    (ROOT / "tests" / "fixtures" / "llm" / "ftp_bruteforce" / "report.json").read_text()
)


class FakeClient:
    base_url = "http://fake"

    def __init__(
        self, rows: list[dict[str, Any]], runs: list[dict[str, Any]] | None = None
    ) -> None:
        self.rows, self.runs = rows, runs or []

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def investigations(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.rows

    def investigation(self, investigation_id: str) -> dict[str, Any]:
        for r in self.rows:
            if r["investigation_id"] == investigation_id:
                return r
        raise KeyError(investigation_id)

    def evaluation_runs(self) -> list[dict[str, Any]]:
        return self.runs

    def run_file(self, run_id: str) -> dict[str, Any] | None:
        return None


def _run(client: FakeClient, page: str = "Alerts", **state: Any) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["client"] = client
    at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def test_dashboard_renders_empty_state() -> None:
    for page in ("Alerts", "Alert detail", "Evaluation"):
        at = _run(FakeClient([]), page)
        assert not at.exception, at.exception
        assert at.info, page


def test_detail_page_shows_uncertainties() -> None:
    row = {
        "investigation_id": "ftp_bruteforce",
        "alert_id": REPORT["alert_id"],
        "status": "done",
        "verdict": REPORT["verdict"],
        "severity": REPORT["severity"],
        "cost_usd": 0.122,
        "created_at": "2026-10-04T00:00:00Z",
        "report": REPORT,
    }
    at = _run(FakeClient([row]), "Alert detail", selected="ftp_bruteforce")
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "true_positive" in text and "T1110" in text
    assert REPORT["uncertainties"][0][:40] in text
    assert any("get_related_events" in c.value for c in at.code)
    alerts = _run(FakeClient([row]), "Alerts")
    assert not alerts.exception and alerts.dataframe


def test_evaluation_page_lists_runs() -> None:
    runs = [
        {
            "run_id": "baseline-rule-based",
            "investigator": "baseline",
            "cases": 38,
            "composite": 0.884,
        }
    ]
    at = _run(FakeClient([], runs), "Evaluation")
    assert not at.exception and at.dataframe
