from pathlib import Path

import pytest
from sqlalchemy import Engine

from secops.db.session import make_engine, upgrade_to_head


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'events.db'}"


@pytest.fixture
def migrated_engine(db_url: str) -> Engine:
    upgrade_to_head(db_url)
    return make_engine(db_url)
