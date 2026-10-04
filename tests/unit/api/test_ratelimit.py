from __future__ import annotations

from fastapi.testclient import TestClient

from secops.api.app import create_app
from secops.api.ratelimit import TokenBucket
from secops.api.settings import ApiSettings


def test_token_bucket_is_per_key_and_refills() -> None:
    clock = [1000.0]
    bucket = TokenBucket(rate_per_minute=2, now=lambda: clock[0])
    assert bucket.take("a") == (True, 0.0)
    assert bucket.take("a") == (True, 0.0)
    allowed, wait = bucket.take("a")
    assert not allowed and 0 < wait <= 30.0
    assert bucket.take("b") == (True, 0.0)  # another key is unaffected
    clock[0] += 30.0  # one token refilled
    assert bucket.take("a") == (True, 0.0)
    allowed, _ = bucket.take("a")
    assert not allowed


def test_rate_limit_is_per_key_and_sets_retry_after(service, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = ApiSettings(api_keys="k1,k2", rate_limit_per_minute=2)
    app = create_app(service=service, settings=settings)
    with TestClient(app) as client:
        for _ in range(2):
            assert client.get("/model", headers={"X-API-Key": "k1"}).status_code == 200
        r = client.get("/model", headers={"X-API-Key": "k1"})
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
        assert client.get("/model", headers={"X-API-Key": "k2"}).status_code == 200
        for _ in range(5):
            assert client.get("/health").status_code == 200  # never limited
        assert client.get("/model", headers={"X-API-Key": "wrong"}).status_code == 401
