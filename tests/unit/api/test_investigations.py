"""Agent endpoints with the recorded ftp_bruteforce scenario replayed (no model, no store)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from secops.api.app import create_app
from secops.api.settings import ApiSettings
from secops.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "llm"
KEY = "k1"


@pytest.fixture
def client(service: Any, full_registry: ToolRegistry, tmp_path: Path) -> TestClient:
    settings = ApiSettings(
        api_keys=KEY,
        rate_limit_per_minute=0,
        agent_mode="replay",
        agent_fixture_root=FIXTURES,
        agent_scenario="ftp_bruteforce",
        evaluation_runs_dir=ROOT / "evaluation" / "runs",
    )
    app = create_app(
        service=service,
        settings=settings,
        registry=full_registry,
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
    )
    with TestClient(app) as c:
        yield c


def _alert() -> dict[str, Any]:
    return json.loads((FIXTURES / "ftp_bruteforce" / "alert.json").read_text())


def test_post_then_get_investigation_replays_the_recorded_scenario(client: TestClient) -> None:
    r = client.post("/investigations", json={"alert": _alert()}, headers={"X-API-Key": KEY})
    assert r.status_code == 202, r.text
    inv_id = r.json()["investigation_id"]
    assert len(inv_id) == 32  # uuid, even in replay mode: a re-run must not collide on the key
    r = client.get(f"/investigations/{inv_id}", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done" and body["verdict"] == "true_positive"
    assert body["report"]["severity"] == "high" and body["cost_usd"] > 0
    r = client.get("/investigations", headers={"X-API-Key": KEY})
    assert r.status_code == 200 and r.json()[0]["investigation_id"] == inv_id
    assert client.get("/investigations/nope", headers={"X-API-Key": KEY}).status_code == 404
    assert client.post("/investigations", json={"alert": _alert()}).status_code == 401


def test_post_investigation_is_idempotent_per_alert(client: TestClient) -> None:
    manager = client.app.state.investigations  # type: ignore[attr-defined]
    alert = _alert()
    from secops.schemas.alert import Alert

    first, started = manager.submit(Alert.model_validate(alert))
    assert started
    second, started_again = manager.submit(Alert.model_validate(alert))
    assert not started_again and second.investigation_id == first.investigation_id
    assert second.status == "queued"


def test_background_failure_is_recorded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import secops.api.investigations as mod

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(mod, "run_investigation", boom)
    r = client.post("/investigations", json={"alert": _alert()}, headers={"X-API-Key": KEY})
    assert r.status_code == 202
    inv_id = r.json()["investigation_id"]
    body = client.get(f"/investigations/{inv_id}", headers={"X-API-Key": KEY}).json()
    assert body["status"] == "failed" and "RuntimeError" in body["error"]
    assert "exploded" not in json.dumps(body["report"]) if body.get("report") else True
    # the alert is no longer marked as running, so a retry can start
    assert (
        client.post(
            "/investigations", json={"alert": _alert()}, headers={"X-API-Key": KEY}
        ).status_code
        == 202
    )


def test_evaluation_runs_listing(client: TestClient) -> None:
    r = client.get("/evaluation/runs", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    runs = {x["run_id"]: x for x in r.json()}
    assert "baseline-rule-based" in runs
    assert (
        runs["baseline-rule-based"]["cases"] == 38
        and runs["baseline-rule-based"]["cost_total_usd"] == 0.0
    )


def test_investigations_can_be_disabled(service: Any, tmp_path: Path) -> None:
    app = create_app(
        service=service,
        settings=ApiSettings(api_keys=KEY, rate_limit_per_minute=0, investigations_enabled=False),
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
    )
    with TestClient(app) as c:
        r = c.post("/investigations", json={"alert": _alert()}, headers={"X-API-Key": KEY})
        assert r.status_code == 503


def test_tracer_is_wired_through_the_api(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import secops.api.investigations as mod
    from secops.observability import RecordingTracer

    tracer = RecordingTracer()
    monkeypatch.setattr(mod, "get_tracer", lambda: tracer)
    r = client.post("/investigations", json={"alert": _alert()}, headers={"X-API-Key": KEY})
    assert r.status_code == 202
    types = [e["type"] for e in tracer.events]
    assert types[0] == "start" and types[-1] == "end" and "llm" in types and "tool" in types
    assert tracer.events[-1]["verdict"] == "true_positive"


def test_second_post_after_completion_starts_a_new_investigation(client: TestClient) -> None:
    first = client.post(
        "/investigations", json={"alert": _alert()}, headers={"X-API-Key": KEY}
    ).json()
    second = client.post(
        "/investigations", json={"alert": _alert()}, headers={"X-API-Key": KEY}
    ).json()
    assert first["investigation_id"] != second["investigation_id"]
    body = client.get(
        f"/investigations/{second['investigation_id']}", headers={"X-API-Key": KEY}
    ).json()
    assert body["status"] == "done"
