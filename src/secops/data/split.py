"""Chronological split assignment and leak checks."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

from secops.data import schema as s


class SplitStrategy(StrEnum):
    CHRONO_WITHIN_GROUP = "chrono_within_group"
    HELDOUT_FRIDAY = "heldout_friday"


SPLIT_COLUMN: dict[SplitStrategy, str] = {
    SplitStrategy.CHRONO_WITHIN_GROUP: "split_chrono",
    SplitStrategy.HELDOUT_FRIDAY: "split_heldout",
}


class LeakError(ValueError):
    """A val/test row precedes a train row within the same group."""


def _time_quantile_within_group(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    """Position of each row inside its group after sorting by time: (rank+1)/size in (0, 1]."""
    ordered = df.sort_values(["Timestamp", "id"])
    g = ordered.groupby(list(group_cols), observed=True)
    rank = g.cumcount()
    size = g["id"].transform("size")
    return ((rank + 1) / size).reindex(df.index)


def assign_chrono_within_group(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    group_cols: Sequence[str] = ("day", "label"),
) -> pd.Series:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("fractions must be in (0,1) and sum to < 1")
    q = _time_quantile_within_group(df, group_cols)
    out = np.where(q <= train_frac, "train", np.where(q <= train_frac + val_frac, "val", "test"))
    return pd.Series(out, index=df.index, name="split")


def assign_heldout_day(
    df: pd.DataFrame, test_day: str = "friday", val_frac: float = 0.15
) -> pd.Series:
    if test_day not in s.DAY_ORDER:
        raise ValueError(f"unknown day {test_day}")
    is_test = (df["day"].astype(str) == test_day).to_numpy()
    rest = df.loc[~is_test]
    q = _time_quantile_within_group(rest, ("day", "label"))
    out = pd.Series("test", index=df.index, name="split")
    out.loc[rest.index] = np.where(q <= 1 - val_frac, "train", "val")
    return out


def check_no_time_leak(
    df: pd.DataFrame, split: pd.Series, group_cols: Sequence[str] = ("day", "label")
) -> None:
    d = df[[*group_cols, "Timestamp"]].assign(split=split.to_numpy())
    agg = d.groupby([*group_cols, "split"], observed=True)["Timestamp"].agg(["min", "max"])
    problems: list[str] = []
    n_group = len(group_cols)
    for key, grp in agg.groupby(level=list(range(n_group))):
        by_split = grp.droplevel(list(range(n_group)))
        tr_max = by_split["max"].get("train")
        va_max = by_split["max"].get("val")
        for later in ("val", "test"):
            lo = by_split["min"].get(later)
            if tr_max is not None and lo is not None and lo < tr_max:
                problems.append(f"{key}: {later} starts {lo} before train ends {tr_max}")
        te_min = by_split["min"].get("test")
        if va_max is not None and te_min is not None and te_min < va_max:
            problems.append(f"{key}: test starts {te_min} before val ends {va_max}")
    if problems:
        raise LeakError("; ".join(problems[:10]))


def split_report(df: pd.DataFrame, split: pd.Series, name: str) -> dict[str, Any]:
    d = df.assign(split=split.to_numpy())
    try:
        check_no_time_leak(df, split)
        leak = "ok"
    except LeakError as e:
        leak = f"LEAK: {e}"
    counts = d.groupby(["split", "family"], observed=True).size()
    by_split_family: dict[str, dict[str, int]] = {sp: {} for sp in ("train", "val", "test")}
    for key, n in counts.items():
        sp, fam = cast(tuple[str, str], key)
        by_split_family[str(sp)][str(fam)] = int(n)
    ranges = d.groupby(["split", "day"], observed=True)["Timestamp"].agg(["min", "max"])
    active_idle_cols = [c for c in s.FEATURE_COLS if c.startswith(("Active", "Idle"))]
    active_idle_max_seconds = float(np.nanmax(df[active_idle_cols].to_numpy()) / 1e6)
    return {
        "name": name,
        "rows": int(len(df)),
        "leak_check": leak,
        "counts_by_split": {str(k): int(v) for k, v in d["split"].value_counts().items()},
        "counts_by_split_family": by_split_family,
        "time_ranges": {
            "/".join(map(str, cast(tuple[str, str], key))): {
                "min": str(r["min"]),
                "max": str(r["max"]),
            }
            for key, r in ranges.iterrows()
        },
        "active_idle_max_seconds": active_idle_max_seconds,
    }
