"""Read raw day CSVs into typed frames. The header must match the improved layout exactly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from secops.data import schema as s


class SchemaError(ValueError):
    """Raised when a CSV header does not match RAW_COLUMNS."""


OUTPUT_COLUMNS: list[str] = [
    "id",
    "day",
    *s.META_COLS,
    *s.FEATURE_COLS,
    s.LABEL_COL,
    s.ATTEMPTED_COL,
]


def _validate_header(path: Path) -> None:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in s.RAW_COLUMNS if c not in header]
    unexpected = [c for c in header if c not in s.RAW_COLUMNS]
    if missing or unexpected or header != s.RAW_COLUMNS:
        raise SchemaError(
            f"{path.name}: header mismatch. missing={missing} unexpected={unexpected} "
            f"ordered_match={header == s.RAW_COLUMNS}"
        )


def read_day(path: Path, day: str) -> pd.DataFrame:
    if day not in s.DAY_ORDER:
        raise ValueError(f"unknown day {day!r}; expected one of {s.DAY_ORDER}")
    _validate_header(path)
    dtypes: dict[str, str] = {c: "float32" for c in s.FEATURE_COLS}
    dtypes.update(
        {
            "id": "int64",
            "Src IP": "category",
            "Dst IP": "category",
            "Src Port": "int32",
            "Dst Port": "int32",
            s.LABEL_COL: "category",
            s.ATTEMPTED_COL: "int8",
        }
    )
    usecols = [c for c in s.RAW_COLUMNS if c != "Flow ID"]
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes, low_memory=False)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["day"] = pd.Categorical([day] * len(df), categories=s.DAY_ORDER)
    for c in s.FEATURE_COLS:
        if df[c].dtype != np.float32:
            df[c] = df[c].astype(np.float32)
    return df[OUTPUT_COLUMNS]


def read_all(
    raw_dir: Path, days: list[str] | None = None, subdir: str = "improved"
) -> pd.DataFrame:
    days = days or s.DAY_ORDER
    frames = [read_day(raw_dir / subdir / s.DAY_FILES[d], d) for d in days]
    out = pd.concat(frames, ignore_index=True)
    out["day"] = pd.Categorical(out["day"].astype(str), categories=days, ordered=True)
    out[s.LABEL_COL] = out[s.LABEL_COL].astype(str).astype("category")
    return out
