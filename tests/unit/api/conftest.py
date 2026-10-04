from pathlib import Path

import pandas as pd
import pytest

from secops.api.detector import DetectorService
from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all
from secops.data.schema import FEATURE_COLS
from secops.schemas.flow import PredictRequest
from tests.fixtures.make_bundle import make_fixture_bundles
from tests.unit.agent.conftest import full_registry, registry  # noqa: F401


@pytest.fixture(scope="session")
def bundle_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return make_fixture_bundles(tmp_path_factory.mktemp("bundles"))


@pytest.fixture(scope="session")
def service(bundle_dirs: tuple[Path, Path]) -> DetectorService:
    det, fam = bundle_dirs
    return DetectorService.from_dirs(det, fam, k=5)


@pytest.fixture(scope="session")
def fixture_flows(fixture_dir: Path) -> pd.DataFrame:
    df, _ = clean(read_all(fixture_dir, subdir=""), AttemptedPolicy.RELABEL_BENIGN)
    return df


def to_request(row: pd.Series, event_id: str | None = None) -> PredictRequest:
    feats = {c: (None if pd.isna(row[c]) else float(row[c])) for c in FEATURE_COLS}
    return PredictRequest(
        event_id=event_id,
        metadata={
            "timestamp": row["Timestamp"].isoformat(),
            "source_ip": str(row["Src IP"]),
            "destination_ip": str(row["Dst IP"]),
            "source_port": int(row["Src Port"]),
            "destination_port": int(row["Dst Port"]),
            "protocol": int(row["Protocol"]),
        },
        features=feats,
    )
