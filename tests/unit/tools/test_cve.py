import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from secops.schemas.tools import MAX_TEXT_CHARS, CveLookupInput
from secops.tools.cve import CveTools
from secops.tools.nvd import FetchResult, NvdClient, NvdError, RateLimiter

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nvd"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class FixtureFetcher:
    """Serves captured NVD responses; records calls so cache behaviour can be asserted."""

    def __init__(self, fail_with: Exception | None = None, status: int | None = None) -> None:
        self.calls: list[dict[str, str]] = []
        self.fail_with = fail_with
        self.status = status

    def __call__(self, params: dict[str, str]) -> FetchResult:
        self.calls.append(params)
        if self.fail_with:
            raise self.fail_with
        if self.status is not None:
            return FetchResult(status_code=self.status, body=None)
        if "cveId" in params:
            fx = _load(params["cveId"].lower())
        else:
            fx = _load("keyword-slowloris")
        return FetchResult(status_code=int(fx["status_code"]), body=fx["body"])


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


@pytest.fixture
def client(tmp_path: Path) -> tuple[NvdClient, FixtureFetcher]:
    fetcher = FixtureFetcher()
    return NvdClient(fetcher=fetcher, cache_dir=tmp_path / "nvd"), fetcher


def test_lookup_by_id_parses_cvss_and_truncates(client: tuple[NvdClient, FixtureFetcher]) -> None:
    nvd, _ = client
    tools = CveTools(nvd)
    r = tools.lookup_cve(CveLookupInput(cve_id="CVE-2014-0160"))
    assert r.status == "found" and len(r.records) == 1 and r.cached is False
    rec = r.records[0]
    assert (
        rec.cve_id == "CVE-2014-0160"
        and rec.cvss_v3_score == 7.5
        and rec.cvss_v3_severity == "HIGH"
    )
    assert rec.description.startswith("The (1) TLS and (2) DTLS implementations in OpenSSL")
    assert len(rec.description) <= MAX_TEXT_CHARS and len(rec.references) <= 5
    assert rec.published.year == 2014 and rec.untrusted_text is True
    assert r.source == "nvd"


def test_lookup_by_keyword_caps_results(client: tuple[NvdClient, FixtureFetcher]) -> None:
    nvd, _ = client
    r = CveTools(nvd).lookup_cve(CveLookupInput(keyword="slowloris", max_results=3))
    assert r.status == "found" and 1 <= len(r.records) <= 3
    assert any(rec.cve_id == "CVE-2007-6750" for rec in r.records)


def test_cve_lookup_unavailable_and_not_found(tmp_path: Path) -> None:
    missing = CveTools(NvdClient(fetcher=FixtureFetcher(), cache_dir=tmp_path / "a"))
    r = missing.lookup_cve(CveLookupInput(cve_id="CVE-9999-99999"))
    assert r.status == "not_found" and r.records == []

    down = CveTools(
        NvdClient(fetcher=FixtureFetcher(fail_with=TimeoutError("boom")), cache_dir=tmp_path / "b")
    )
    r = down.lookup_cve(CveLookupInput(cve_id="CVE-2014-0160"))
    assert r.status == "unavailable" and r.records == [] and r.cached is False

    limited = CveTools(NvdClient(fetcher=FixtureFetcher(status=403), cache_dir=tmp_path / "c"))
    r = limited.lookup_cve(CveLookupInput(cve_id="CVE-2014-0160"))
    assert r.status == "unavailable"


def test_cache_hit_skips_fetcher_and_serves_when_down(tmp_path: Path) -> None:
    fetcher = FixtureFetcher()
    nvd = NvdClient(fetcher=fetcher, cache_dir=tmp_path / "nvd")
    first = CveTools(nvd).lookup_cve(CveLookupInput(cve_id="CVE-2007-6750"))
    assert first.cached is False and len(fetcher.calls) == 1
    second = CveTools(nvd).lookup_cve(CveLookupInput(cve_id="CVE-2007-6750"))
    assert second.cached is True and len(fetcher.calls) == 1
    assert second.records[0].cve_id == "CVE-2007-6750"
    # same cache dir, fetcher now down: the cached entry is still served
    down = NvdClient(fetcher=FixtureFetcher(fail_with=OSError("down")), cache_dir=tmp_path / "nvd")
    r = CveTools(down).lookup_cve(CveLookupInput(cve_id="CVE-2007-6750"))
    assert r.status == "found" and r.cached is True


def test_rate_limiter_spaces_requests_with_fake_clock() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_requests=5, per_seconds=30, clock=clock.time, sleep=clock.sleep)
    for _ in range(5):
        limiter.acquire()
    assert clock.slept == []
    limiter.acquire()  # sixth call inside the window must wait until the first expires
    assert len(clock.slept) == 1 and 29.0 <= clock.slept[0] <= 30.0


def test_fetch_error_types() -> None:
    assert issubclass(NvdError, Exception)


def test_cve_input_bounds() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        CveLookupInput()
    with pytest.raises(ValidationError, match="exactly one"):
        CveLookupInput(cve_id="CVE-2014-0160", keyword="x")
    with pytest.raises(ValidationError):
        CveLookupInput(cve_id="not-a-cve")
    with pytest.raises(ValidationError):
        CveLookupInput(keyword="x" * 101)
    with pytest.raises(ValidationError):
        CveLookupInput(keyword="x", max_results=6)
