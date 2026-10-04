from pathlib import Path

import pytest
from sqlalchemy import Engine

from secops.tools.nvd import NvdClient
from secops.tools.registry import ToolRegistry, build_registry
from tests.unit.tools.test_cve import FixtureFetcher

SUBSET = Path(__file__).resolve().parents[2] / "fixtures" / "attack" / "index_subset.json"


@pytest.fixture(scope="session")
def registry(event_engine: Engine, tmp_path_factory: pytest.TempPathFactory) -> ToolRegistry:
    return build_registry(
        engine=event_engine,
        attack_index_path=SUBSET,
        nvd_client=NvdClient(fetcher=FixtureFetcher(), cache_dir=tmp_path_factory.mktemp("nvd")),
        include_detector=False,
    )
