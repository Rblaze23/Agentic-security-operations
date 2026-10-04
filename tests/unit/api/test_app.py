import json
import statistics
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from secops.api.app import create_app
from secops.api.detector import DetectorService
from secops.api.settings import ApiSettings
from secops.detection.bundle import BundleError
from secops.schemas.flow import BATCH_LIMIT
from tests.unit.api.conftest import to_request

KEY = "test-key-1"
HEADERS = {"X-API-Key": KEY}


@pytest.fixture(scope="module")
def client(service: DetectorService, tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    settings = ApiSettings(api_keys=f"{KEY},other-key", model_dir=tmp_path_factory.mktemp("unused"))
    return TestClient(create_app(service=service, settings=settings))


@pytest.fixture(scope="module")
def payload(fixture_flows: pd.DataFrame) -> dict[str, object]:
    return to_request(fixture_flows.iloc[0], event_id="evt-0").model_dump(mode="json")


@pytest.fixture(scope="module")
def alert_payload(service: DetectorService, fixture_flows: pd.DataFrame) -> dict[str, object]:
    for _, row in fixture_flows.iterrows():
        req = to_request(row, event_id="evt-alert")
        if service.predict([req])[0].is_alert:
            return req.model_dump(mode="json")
    raise AssertionError("fixture has no alert row")


def test_health_is_open_and_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["detector_loaded"] and body["family_loaded"]
    assert body["bundle_versions"] == {"detector": 0, "family": 0}
    assert "X-Request-ID" in r.headers


def test_auth_required_on_predict_only(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/predict", json=payload).status_code == 401
    r = client.post("/predict", json=payload, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401 and r.headers["WWW-Authenticate"].startswith("ApiKey")
    assert client.post("/predict/batch", json={"items": [payload]}).status_code == 401
    assert client.get("/model").status_code == 401
    assert client.get("/health").status_code == 200
    assert (
        client.post("/predict", json=payload, headers={"X-API-Key": "other-key"}).status_code == 200
    )


def test_predict_happy_path_and_alert_object(
    client: TestClient, alert_payload: dict[str, object]
) -> None:
    r = client.post("/predict", json=alert_payload, headers=HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    p = body["prediction"]
    assert p["event_id"] == "evt-alert" and p["is_alert"] is True
    assert 0.0 <= p["attack_probability"] <= 1.0 and len(p["top_contributions"]) == 5
    assert p["predicted_family"] is not None and p["model_name"] == "secops-detector-fixture"
    a = body["alert"]
    assert a is not None and a["event_id"] == "evt-alert" and a["status"] == "new"
    assert a["prediction"]["prediction_id"] == p["prediction_id"]
    assert set(a["key_features"]) == {c["feature"] for c in p["top_contributions"]}
    assert a["metadata"]["destination_ip"] == alert_payload["metadata"]["destination_ip"]  # type: ignore[index]


def test_predict_non_alert_has_no_alert_object(
    client: TestClient, service: DetectorService, fixture_flows: pd.DataFrame
) -> None:
    for _, row in fixture_flows.iterrows():
        req = to_request(row)
        if not service.predict([req])[0].is_alert:
            r = client.post("/predict", json=req.model_dump(mode="json"), headers=HEADERS)
            assert r.status_code == 200 and r.json()["alert"] is None
            assert r.json()["prediction"]["predicted_family"] is None
            return
    raise AssertionError("fixture has no benign row")


def test_validation_errors_are_422(client: TestClient, payload: dict[str, object]) -> None:
    bad = json.loads(json.dumps(payload))
    del bad["features"]["Flow Duration"]
    r = client.post("/predict", json=bad, headers=HEADERS)
    assert r.status_code == 422 and "Flow Duration" in r.text
    extra = json.loads(json.dumps(payload))
    extra["surprise"] = 1
    assert client.post("/predict", json=extra, headers=HEADERS).status_code == 422
    too_many = {"items": [payload] * (BATCH_LIMIT + 1)}
    assert client.post("/predict/batch", json=too_many, headers=HEADERS).status_code == 422
    assert client.post("/predict/batch", json={"items": []}, headers=HEADERS).status_code == 422


def test_predict_rejects_non_json_numbers(client: TestClient, payload: dict[str, object]) -> None:
    raw = json.dumps(payload).replace('"Flow Duration": 1.0', '"Flow Duration": NaN')
    raw = json.dumps(payload)
    body = json.loads(raw)
    body["features"]["Flow Duration"] = "NaN-placeholder"
    text = json.dumps(body).replace('"NaN-placeholder"', "NaN")
    r = client.post(
        "/predict", content=text, headers={**HEADERS, "Content-Type": "application/json"}
    )
    assert r.status_code == 422
    text_inf = json.dumps(body).replace('"NaN-placeholder"', "Infinity")
    r = client.post(
        "/predict", content=text_inf, headers={**HEADERS, "Content-Type": "application/json"}
    )
    assert r.status_code == 422


def test_batch_limit_and_ids(client: TestClient, payload: dict[str, object]) -> None:
    r = client.post("/predict/batch", json={"items": [payload] * BATCH_LIMIT}, headers=HEADERS)
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == BATCH_LIMIT
    assert len({p["prediction_id"] for p in preds}) == BATCH_LIMIT
    assert len({p["attack_probability"] for p in preds}) == 1
    alerts = r.json()["alerts"]
    assert len(alerts) == sum(p["is_alert"] for p in preds)


def test_model_info_matches_manifests(client: TestClient, service: DetectorService) -> None:
    r = client.get("/model", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["detector"]["run_id"] == service.detector.manifest.run_id
    assert body["detector"]["model_kind"] == "lightgbm" and body["threshold"] == service.threshold
    assert body["family"]["model_name"] == "secops-family-fixture" and body["top_k"] == 5


def test_not_loaded_reports_starting_and_503(tmp_path: Path) -> None:
    app = create_app(service=None, settings=ApiSettings(api_keys=KEY, model_dir=tmp_path))
    c = TestClient(app)  # no lifespan: nothing loaded
    h = c.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "starting"
    assert c.get("/model", headers=HEADERS).status_code == 503
    assert c.post("/predict", json={"features": {"x": 1}}, headers=HEADERS).status_code == 503


def test_lifespan_loads_bundles_from_model_dir(
    bundle_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    det, fam = bundle_dirs
    settings = ApiSettings(api_keys=KEY, model_dir=det.parent)
    with TestClient(create_app(settings=settings)) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert (
            c.get("/model", headers=HEADERS).json()["family"]["model_name"]
            == "secops-family-fixture"
        )


def test_lifespan_fails_fast_on_missing_bundle(tmp_path: Path) -> None:
    settings = ApiSettings(api_keys=KEY, model_dir=tmp_path / "nowhere")
    with pytest.raises(BundleError):
        with TestClient(create_app(settings=settings)):
            pass


def test_no_keys_configured_rejects_everything_protected(
    service: DetectorService, payload: dict[str, object], tmp_path: Path
) -> None:
    c = TestClient(
        create_app(service=service, settings=ApiSettings(api_keys="", model_dir=tmp_path))
    )
    assert c.get("/health").status_code == 200
    assert c.post("/predict", json=payload, headers=HEADERS).status_code == 401


def test_single_request_latency(client: TestClient, payload: dict[str, object]) -> None:
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        assert client.post("/predict", json=payload, headers=HEADERS).status_code == 200
        times.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(0.95 * len(times)) - 1]
    print(f"\nsingle /predict latency over 100 calls: p50={p50:.1f} ms p95={p95:.1f} ms")
    assert p95 < 5000


def test_non_ascii_api_key_is_401_not_500(client: TestClient, payload: dict[str, object]) -> None:
    r = client.post("/predict", json=payload, headers=[(b"X-API-Key", b"k\xe9")])
    assert r.status_code == 401


def test_internal_value_error_is_500_without_message(
    service: DetectorService,
    payload: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(service=service, settings=ApiSettings(api_keys=KEY, model_dir=tmp_path))
    c = TestClient(app, raise_server_exceptions=False)

    def boom(X: object) -> object:
        raise ValueError("internal: /app/secret/path")

    monkeypatch.setattr(service.detector.estimator, "predict_proba", boom)
    r = c.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 500
    assert "secret" not in r.text and r.json()["detail"] == "internal error"
    assert "X-Request-ID" in r.headers


def test_missing_family_bundle_reports_degraded(
    bundle_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    import shutil

    det, _ = bundle_dirs
    only_detector = tmp_path / "models"
    shutil.copytree(det, only_detector / "detector")
    with TestClient(create_app(settings=ApiSettings(api_keys=KEY, model_dir=only_detector))) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["detector_loaded"] and not h["family_loaded"]
        r = c.post("/predict", json=client_payload_from(only_detector), headers=HEADERS)
        assert r.status_code == 200 and r.json()["prediction"]["predicted_family"] is None


def client_payload_from(_: Path) -> dict[str, object]:
    from secops.data.clean import AttemptedPolicy, clean
    from secops.data.ingest import read_all
    from tests.conftest import FIXTURE_DIR

    df, _ = clean(read_all(FIXTURE_DIR, subdir=""), AttemptedPolicy.RELABEL_BENIGN)
    return to_request(df.iloc[0]).model_dump(mode="json")


def test_default_host_is_loopback() -> None:
    assert ApiSettings(api_keys=KEY).host == "127.0.0.1"


def test_unauthenticated_requests_are_rejected_before_the_body_is_parsed(
    client: TestClient, payload: dict[str, object]
) -> None:
    too_many = {"items": [payload] * (BATCH_LIMIT + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 401  # not 422
    assert (
        client.post(
            "/predict", content=b"{not json", headers={"Content-Type": "application/json"}
        ).status_code
        == 401
    )


def test_oversized_body_is_413(client: TestClient) -> None:
    r = client.post(
        "/predict",
        content=b"{}",
        headers={
            **HEADERS,
            "Content-Type": "application/json",
            "Content-Length": str(50 * 1024 * 1024),
        },
    )
    assert r.status_code == 413
