from pathlib import Path

import pandas as pd
import pytest

from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all
from secops.data.split import (
    LeakError,
    assign_chrono_within_group,
    assign_heldout_day,
    check_no_time_leak,
    split_report,
)


@pytest.fixture
def cleaned(fixture_dir: Path) -> pd.DataFrame:
    df, _ = clean(read_all(fixture_dir, subdir=""), AttemptedPolicy.RELABEL_BENIGN)
    return df


def test_chrono_split_fractions_and_coverage(cleaned: pd.DataFrame) -> None:
    split = assign_chrono_within_group(cleaned)
    assert set(split.unique()) <= {"train", "val", "test"}
    frac = split.value_counts(normalize=True)
    assert 0.55 < frac["train"] < 0.85
    d = cleaned.assign(split=split)
    for _, grp in d.groupby(["day", "label"], observed=True):
        if len(grp) >= 7:
            assert set(grp["split"]) == {"train", "val", "test"}


def test_chrono_split_is_time_ordered_within_group(cleaned: pd.DataFrame) -> None:
    split = assign_chrono_within_group(cleaned)
    check_no_time_leak(cleaned, split)  # must not raise
    d = cleaned.assign(split=split)
    for _, grp in d.groupby(["day", "label"], observed=True):
        if {"train", "test"} <= set(grp["split"]):
            tr = grp.loc[grp.split == "train", "Timestamp"].max()
            te = grp.loc[grp.split == "test", "Timestamp"].min()
            assert tr <= te


def test_check_no_time_leak_raises(cleaned: pd.DataFrame) -> None:
    split = assign_chrono_within_group(cleaned).copy()
    d = cleaned.assign(split=split)
    earliest = d[(d.day == "monday") & (d.label == "BENIGN")].sort_values("Timestamp").index[0]
    split.loc[earliest] = "test"
    with pytest.raises(LeakError):
        check_no_time_leak(cleaned, split)


def test_heldout_day_puts_all_friday_in_test(cleaned: pd.DataFrame) -> None:
    split = assign_heldout_day(cleaned, test_day="friday")
    assert (split[cleaned.day == "friday"] == "test").all()
    assert (split[cleaned.day != "friday"] != "test").all()
    assert set(split[cleaned.day != "friday"].unique()) == {"train", "val"}


def test_invalid_fractions_rejected(cleaned: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        assign_chrono_within_group(cleaned, train_frac=0.9, val_frac=0.2)


def test_split_report_has_counts_and_ranges(cleaned: pd.DataFrame) -> None:
    split = assign_chrono_within_group(cleaned)
    rep = split_report(cleaned, split, "chrono_within_group")
    assert rep["name"] == "chrono_within_group"
    assert rep["rows"] == len(cleaned)
    assert set(rep["counts_by_split_family"]) == {"train", "val", "test"}
    assert rep["leak_check"] == "ok"
    assert rep["active_idle_max_seconds"] < 86_400
