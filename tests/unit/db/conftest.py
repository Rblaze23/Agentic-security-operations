from pathlib import Path

import pytest
from sqlalchemy import Engine

from secops.data.build import build
from secops.data.clean import AttemptedPolicy
from secops.db.session import make_engine, upgrade_to_head


@pytest.fixture(scope="session")
def fixture_parquet(fixture_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("built")
    return build(
        fixture_dir, tmp / "processed", tmp / "reports", AttemptedPolicy.RELABEL_BENIGN, subdir=""
    )


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'events.db'}"


@pytest.fixture
def migrated_engine(db_url: str) -> Engine:
    upgrade_to_head(db_url)
    return make_engine(db_url)
