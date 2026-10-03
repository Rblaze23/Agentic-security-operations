from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mini_cicids"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR
