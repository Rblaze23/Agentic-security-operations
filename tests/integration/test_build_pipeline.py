import json
from pathlib import Path

import pandas as pd
import pytest

from secops.data import schema as s
from secops.data.build import build
from secops.data.clean import AttemptedPolicy

pytestmark = pytest.mark.integration


def test_build_writes_parquet_and_reports(fixture_dir: Path, tmp_path: Path) -> None:
    out = build(
        fixture_dir,
        tmp_path / "processed",
        tmp_path / "reports",
        AttemptedPolicy.RELABEL_BENIGN,
        subdir="",
    )
    df = pd.read_parquet(out)
    expected = [
        "id",
        "day",
        *s.META_COLS,
        *s.FEATURE_COLS,
        "label_raw",
        "label",
        "family",
        "is_attack",
        "split_chrono",
        "split_heldout",
    ]
    for c in expected:
        assert c in df.columns, c
    rep = json.loads((tmp_path / "reports" / "relabel_benign" / "split_report.json").read_text())
    assert rep["chrono_within_group"]["leak_check"] == "ok"
    assert rep["heldout_friday"]["leak_check"] == "ok"
    cl = json.loads((tmp_path / "reports" / "relabel_benign" / "cleaning_report.json").read_text())
    assert cl["rows_out"] == len(df)
