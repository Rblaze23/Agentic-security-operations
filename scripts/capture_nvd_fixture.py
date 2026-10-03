"""Capture NVD API 2.0 responses as test fixtures (run rarely; respects the 5-per-30-s limit).

Usage: uv run python scripts/capture_nvd_fixture.py
Writes tests/fixtures/nvd/<name>.json with the raw response and a README with the capture date.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "nvd"
BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
QUERIES: dict[str, dict[str, str]] = {
    "cve-2014-0160": {"cveId": "CVE-2014-0160"},
    "cve-2007-6750": {"cveId": "CVE-2007-6750"},
    "cve-9999-99999": {"cveId": "CVE-9999-99999"},
    "keyword-slowloris": {"keywordSearch": "slowloris", "resultsPerPage": "5"},
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    captured = datetime.now(UTC).isoformat()
    for name, params in QUERIES.items():
        r = requests.get(
            BASE, params=params, timeout=30, headers={"User-Agent": "secops-fixture-capture"}
        )
        payload = {"status_code": r.status_code, "params": params, "captured": captured}
        try:
            payload["body"] = r.json()
        except ValueError:
            payload["body"] = None
            payload["text"] = r.text[:500]
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1))
        n = (
            (payload["body"] or {}).get("totalResults")
            if isinstance(payload["body"], dict)
            else None
        )
        print(f"{name}: HTTP {r.status_code}, totalResults={n}")
        time.sleep(7)  # stay under 5 requests / 30 s without an API key
    (OUT / "README.md").write_text(
        "# NVD fixtures\n\nRaw responses of the NVD API 2.0 (`"
        + BASE
        + "`) captured on "
        + captured
        + " by `scripts/capture_nvd_fixture.py`. Used by the CVE tool tests through the "
        "injectable fetcher; never fetched live in tests. NVD data is public domain (NIST); "
        "see https://nvd.nist.gov/developers/terms-of-use.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
