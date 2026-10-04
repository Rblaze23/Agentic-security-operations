"""NVD API 2.0 client: injectable fetcher, on-disk cache with TTL, token-bucket rate limiter,
and a parser from the NVD JSON to `CveRecord`. Network errors never escape as exceptions to the
tool; they become `NvdError`, which the tool maps to status="unavailable"."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from secops.schemas.tools import MAX_REFERENCES, MAX_TEXT_CHARS, CveRecord

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
DEFAULT_TTL_DAYS = 7


log = logging.getLogger(__name__)


class NvdError(Exception):
    """The NVD service could not be used (network, timeout, non-200 status)."""


@dataclass
class FetchResult:
    status_code: int
    body: dict[str, Any] | None


Fetcher = Callable[[dict[str, str]], FetchResult]


def http_fetcher(api_key: str | None = None, timeout: float = 10.0) -> Fetcher:
    """Real HTTP fetcher. `api_key` (NVD_API_KEY) raises the rate limit from 5 to 50 per 30 s."""

    def fetch(params: dict[str, str]) -> FetchResult:
        headers = {"User-Agent": "secops-detection/0.1"}
        if api_key:
            headers["apiKey"] = api_key
        r = requests.get(NVD_BASE_URL, params=params, headers=headers, timeout=timeout)
        try:
            body = r.json() if r.content else None
        except ValueError:
            body = None
        return FetchResult(status_code=r.status_code, body=body)

    return fetch


class RateLimiter:
    """At most `max_requests` acquisitions per rolling `per_seconds` window; blocks otherwise."""

    def __init__(
        self,
        max_requests: int,
        per_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()

    def acquire(self) -> None:
        now = self._clock()
        while self._stamps and now - self._stamps[0] >= self.per_seconds:
            self._stamps.popleft()
        if len(self._stamps) >= self.max_requests:
            wait = self.per_seconds - (now - self._stamps[0])
            self._sleep(wait)
            now = self._clock()
            while self._stamps and now - self._stamps[0] >= self.per_seconds:
                self._stamps.popleft()
        self._stamps.append(now)


def parse_records(body: dict[str, Any]) -> list[CveRecord]:
    out: list[CveRecord] = []
    for item in body.get("vulnerabilities", []):
        cve = item.get("cve", {})
        metrics = cve.get("metrics", {})
        score: float | None = None
        severity: str | None = None
        for key in ("cvssMetricV31", "cvssMetricV30"):
            entries = metrics.get(key) or []
            if entries:
                data = entries[0].get("cvssData", {})
                score = float(data["baseScore"]) if "baseScore" in data else None
                severity = data.get("baseSeverity")
                break
        descriptions = [
            d.get("value", "") for d in cve.get("descriptions", []) if d.get("lang") == "en"
        ]
        text = " ".join((descriptions[0] if descriptions else "").split())
        if len(text) > MAX_TEXT_CHARS:
            text = text[: MAX_TEXT_CHARS - 1] + "…"
        refs = [r.get("url", "") for r in cve.get("references", []) if r.get("url")][
            :MAX_REFERENCES
        ]
        out.append(
            CveRecord(
                cve_id=str(cve.get("id")),
                published=_parse_ts(cve.get("published")),
                last_modified=_parse_ts(cve.get("lastModified")),
                cvss_v3_score=score,
                cvss_v3_severity=severity,
                description=text,
                references=refs,
            )
        )
    return out


def _parse_ts(value: Any) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=UTC)
    dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class NvdClient:
    def __init__(
        self,
        fetcher: Fetcher | None = None,
        cache_dir: Path | None = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
        limiter: RateLimiter | None = None,
    ) -> None:
        api_key = os.environ.get("NVD_API_KEY")
        self.fetcher = fetcher or http_fetcher(api_key)
        self.cache_dir = cache_dir or Path.home() / ".cache" / "secops" / "nvd"
        self.ttl = timedelta(days=ttl_days)
        self.limiter = limiter or RateLimiter(50 if api_key else 5, 30.0)

    # ---- cache ---------------------------------------------------------------------------
    def _cache_path(self, params: dict[str, str]) -> Path:
        key = hashlib.sha1(  # noqa: S324 - cache file name, not security
            json.dumps(params, sort_keys=True).encode(), usedforsecurity=False
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(
        self, params: dict[str, str], ignore_ttl: bool = False
    ) -> dict[str, Any] | None:
        p = self._cache_path(params)
        if not p.exists():
            return None
        try:
            entry = json.loads(p.read_text())
            age = datetime.now(UTC) - datetime.fromisoformat(entry["stored_at"])
        except (ValueError, KeyError, TypeError) as e:  # corrupt or hand-edited cache file
            log.warning("ignoring unreadable NVD cache entry %s (%s)", p.name, e)
            return None
        if age > self.ttl and not ignore_ttl:
            return None
        body: dict[str, Any] = entry["body"]
        return body

    def _write_cache(self, params: dict[str, str], body: dict[str, Any]) -> None:
        """Best effort: the cache is an optimisation, a read-only data mount must not turn a
        successful NVD answer into a tool error."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(params).write_text(
                json.dumps(
                    {"stored_at": datetime.now(UTC).isoformat(), "params": params, "body": body}
                )
            )
        except OSError as e:
            log.warning("NVD cache not writable (%s); answer served without caching", e)

    # ---- queries -------------------------------------------------------------------------
    def query(self, params: dict[str, str]) -> tuple[dict[str, Any], bool]:
        """Return (body, cached). Raises NvdError when neither network nor cache can answer."""
        cached = self._read_cache(params)
        if cached is not None:
            return cached, True
        try:
            self.limiter.acquire()
            result = self.fetcher(params)
        except Exception as e:  # noqa: BLE001  any transport failure is "unavailable"
            stale = self._read_cache(params, ignore_ttl=True)
            if stale is not None:
                return stale, True
            raise NvdError(f"NVD request failed: {type(e).__name__}: {e}") from e
        if result.status_code != 200 or result.body is None:
            stale = self._read_cache(params, ignore_ttl=True)
            if stale is not None:
                return stale, True
            raise NvdError(f"NVD returned HTTP {result.status_code}")
        self._write_cache(params, result.body)
        return result.body, False

    def by_id(self, cve_id: str) -> tuple[list[CveRecord], bool]:
        body, cached = self.query({"cveId": cve_id.upper()})
        return parse_records(body), cached

    def by_keyword(self, keyword: str, max_results: int) -> tuple[list[CveRecord], bool]:
        body, cached = self.query({"keywordSearch": keyword, "resultsPerPage": str(max_results)})
        return parse_records(body)[:max_results], cached
