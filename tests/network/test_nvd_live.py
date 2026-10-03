"""Opt-in live check of the NVD client: `uv run pytest tests/network -m network`."""

from pathlib import Path

import pytest

from secops.schemas.tools import CveLookupInput
from secops.tools.cve import CveTools
from secops.tools.nvd import NvdClient, http_fetcher

pytestmark = pytest.mark.network


def test_live_heartbleed(tmp_path: Path) -> None:
    nvd = NvdClient(fetcher=http_fetcher(), cache_dir=tmp_path)
    r = CveTools(nvd).lookup_cve(CveLookupInput(cve_id="CVE-2014-0160"))
    assert r.status in {"found", "unavailable"}
    if r.status == "found":
        assert r.records[0].cvss_v3_score == 7.5
