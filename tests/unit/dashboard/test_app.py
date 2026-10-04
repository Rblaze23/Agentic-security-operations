"""The dashboard against a fake client: every page renders offline, the detail page shows a
full report with evidence and uncertainties, live pages degrade with guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
from dashboard import data  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "dashboard" / "app.py"
PAGES = [
    "Overview",
    "Detector",
    "Agent results",
    "Investigations",
    "Investigation detail",
    "Evaluation runs",
]


class DownClient:
    base_url = "http://fake"

    def health(self) -> dict[str, Any]:
        raise ConnectionError("down")

    def investigations(self, limit: int = 100) -> list[dict[str, Any]]:
        raise ConnectionError("down")

    def investigation(self, investigation_id: str) -> dict[str, Any]:
        raise ConnectionError("down")

    def evaluation_runs(self) -> list[dict[str, Any]]:
        raise ConnectionError("down")

    def run_file(self, run_id: str) -> dict[str, Any] | None:
        return None


class UpClient(DownClient):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def investigations(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.rows

    def investigation(self, investigation_id: str) -> dict[str, Any]:
        for r in self.rows:
            if r["investigation_id"] == investigation_id:
                return r
        raise KeyError(investigation_id)


def _run(client: Any, page: str, **state: Any) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["client"] = client
    at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def _text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_offline_without_errors(page: str) -> None:
    at = _run(DownClient(), page)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]  # offline is guidance, never an error box


def test_overview_explains_the_project_and_the_metrics() -> None:
    at = _run(DownClient(), "Overview")
    text = _text(at)
    assert "defensive security platform" in text and "How to read it" in text
    assert at.dataframe, "headline table missing"
    assert "0.921" in str(at.dataframe[0].value.to_dict())
    assert any("not running" in i.value for i in at.info)


def test_results_page_lists_every_golden_case() -> None:
    at = _run(DownClient(), "Agent results")
    frames = at.dataframe
    assert len(frames) >= 2
    per_case = frames[-1].value
    assert len(per_case) == len(data.golden_cases()) == 38
    assert "✅ true_positive" in str(per_case.to_dict())


def test_detail_page_shows_a_recorded_report_with_evidence() -> None:
    at = _run(DownClient(), "Investigation detail", selected="scenario:ftp_bruteforce")
    assert not at.exception, at.exception
    heads = " ".join(h.value for h in at.subheader)
    assert "What the agent did" in heads and "Uncertainties" in heads
    assert "T1110" in _text(at)
    assert any("get_related_events" in c.value for c in at.code)
    assert [m.label for m in at.metric][:2] == ["Verdict", "Severity"]


def test_live_pages_use_the_api_when_it_is_up() -> None:
    import json

    report = json.loads((ROOT / "tests/fixtures/llm/ftp_bruteforce/report.json").read_text())
    row = {
        "investigation_id": "abc",
        "alert_id": report["alert_id"],
        "status": "done",
        "verdict": report["verdict"],
        "severity": report["severity"],
        "cost_usd": 0.12,
        "created_at": "2026-10-04T00:00:00Z",
        "report": report,
    }
    at = _run(UpClient([row]), "Investigations")
    assert not at.exception and len(at.dataframe) == 2  # recorded + live tables
    at = _run(UpClient([row]), "Investigation detail", selected="live:abc")
    assert not at.exception
    assert any(m.value == "true_positive" for m in at.metric)
