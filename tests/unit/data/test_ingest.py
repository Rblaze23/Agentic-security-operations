from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secops.data import schema as s
from secops.data.ingest import SchemaError, read_all, read_day


def test_read_day_returns_expected_columns_and_dtypes(fixture_dir: Path) -> None:
    df = read_day(fixture_dir / "tuesday.csv", "tuesday")
    expected = ["id", "day", *s.META_COLS, *s.FEATURE_COLS, s.LABEL_COL, s.ATTEMPTED_COL]
    assert list(df.columns) == expected
    assert all(df[c].dtype == np.float32 for c in s.FEATURE_COLS)
    assert isinstance(df["Timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(df["Timestamp"].dt.tz) == "UTC"
    assert df["day"].iloc[0] == "tuesday"
    assert (df[s.LABEL_COL] == "FTP-Patator").any()


def test_read_day_rejects_wrong_header(tmp_path: Path) -> None:
    bad = tmp_path / "monday.csv"
    bad.write_text("id,Flow ID,Src IP,Label\n1,x,1.1.1.1,BENIGN\n")
    with pytest.raises(SchemaError) as exc:
        read_day(bad, "monday")
    assert "missing" in str(exc.value) and "Flow Duration" in str(exc.value)


def test_read_day_rejects_unknown_day(fixture_dir: Path) -> None:
    with pytest.raises(ValueError):
        read_day(fixture_dir / "monday.csv", "sunday")


def test_read_all_concatenates_days_in_order(fixture_dir: Path) -> None:
    df = read_all(fixture_dir, days=["monday", "tuesday"], subdir="")
    assert list(df["day"].cat.categories) == ["monday", "tuesday"]
    assert df["day"].cat.codes.is_monotonic_increasing
    assert (df["day"] == "tuesday").sum() > 0


def test_timestamps_are_within_known_capture_window(fixture_dir: Path) -> None:
    df = read_day(fixture_dir / "friday.csv", "friday")
    assert df["Timestamp"].min() >= pd.Timestamp("2017-07-07", tz="UTC")
    assert df["Timestamp"].max() < pd.Timestamp("2017-07-08", tz="UTC")
