"""Shared fixtures: a built fixture dataset, its frame, and a loaded read-only event store."""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine

from secops.data.build import build
from secops.data.clean import AttemptedPolicy
from secops.db.events_loader import load_events
from secops.db.session import make_engine, upgrade_to_head
from secops.detection.features import FEATURE_SPEC_V1


@pytest.fixture(scope="session")
def fixture_parquet(fixture_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("built")
    return build(
        fixture_dir, tmp / "processed", tmp / "reports", AttemptedPolicy.RELABEL_BENIGN, subdir=""
    )


@pytest.fixture(scope="session")
def flows(fixture_parquet: Path) -> pd.DataFrame:
    return pd.read_parquet(fixture_parquet)


@pytest.fixture(scope="session")
def event_engine(fixture_parquet: Path, tmp_path_factory: pytest.TempPathFactory) -> Engine:
    url = f"sqlite:///{tmp_path_factory.mktemp('db') / 'events.db'}"
    upgrade_to_head(url)
    load_events(fixture_parquet, make_engine(url), FEATURE_SPEC_V1)
    return make_engine(url, read_only=True)
