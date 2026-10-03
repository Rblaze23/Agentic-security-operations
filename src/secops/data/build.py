"""End-to-end build: raw CSVs -> cleaned, split Parquet + JSON reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all
from secops.data.manifest import manifest_digest
from secops.data.split import (
    SPLIT_COLUMN,
    SplitStrategy,
    assign_chrono_within_group,
    assign_heldout_day,
    check_no_time_leak,
    split_report,
)

log = logging.getLogger(__name__)


def build(
    raw_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    policy: AttemptedPolicy,
    subdir: str = "improved",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Path:
    log.info("reading raw CSVs from %s", raw_dir / subdir)
    df = read_all(raw_dir, subdir=subdir)
    log.info("cleaning %d rows with policy=%s", len(df), policy)
    df, cleaning = clean(df, policy)
    chrono = assign_chrono_within_group(df, train_frac=train_frac, val_frac=val_frac)
    heldout = assign_heldout_day(df, test_day="friday", val_frac=val_frac)
    check_no_time_leak(df, chrono)
    check_no_time_leak(df, heldout)
    df[SPLIT_COLUMN[SplitStrategy.CHRONO_WITHIN_GROUP]] = chrono.astype("category")
    df[SPLIT_COLUMN[SplitStrategy.HELDOUT_FRIDAY]] = heldout.astype("category")

    out_dir = processed_dir / str(policy)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "flows.parquet"
    df.to_parquet(out, index=False)

    rep_dir = reports_dir / str(policy)
    cleaning.to_json(rep_dir / "cleaning_report.json")
    reports = {
        "manifest_digest": manifest_digest(),
        "attempted_policy": str(policy),
        "chrono_within_group": split_report(df, chrono, "chrono_within_group"),
        "heldout_friday": split_report(df, heldout, "heldout_friday"),
    }
    (rep_dir / "split_report.json").write_text(json.dumps(reports, indent=2))
    log.info("wrote %s (%d rows)", out, len(df))
    return out
