from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mini_cicids"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR
