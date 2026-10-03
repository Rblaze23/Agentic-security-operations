import json
from pathlib import Path

import numpy as np
import pandas as pd

from secops.data import schema as s
from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all


def _df(fixture_dir: Path) -> pd.DataFrame:
    return read_all(fixture_dir, subdir="")


def test_clean_replaces_inf_with_nan(fixture_dir: Path) -> None:
    df = _df(fixture_dir)
    df.loc[df.index[0], "Flow Bytes/s"] = np.float32("inf")
    df.loc[df.index[1], "Flow Packets/s"] = np.float32("-inf")
    out, rep = clean(df, AttemptedPolicy.RELABEL_BENIGN)
    assert rep.inf_cells_replaced == 2
    assert not np.isinf(out[s.FEATURE_COLS].to_numpy()).any()


def test_relabel_benign_policy_turns_attempted_into_benign(fixture_dir: Path) -> None:
    df = _df(fixture_dir)
    n_attempted = df[s.LABEL_COL].astype(str).str.endswith(s.ATTEMPTED_SUFFIX).sum()
    out, rep = clean(df, AttemptedPolicy.RELABEL_BENIGN)
    assert rep.attempted_rows == n_attempted
    assert not out["label"].astype(str).str.endswith(s.ATTEMPTED_SUFFIX).any()
    attempted = out[out["label_raw"].astype(str).str.endswith(s.ATTEMPTED_SUFFIX)]
    assert (attempted["label"] == s.BENIGN_LABEL).all()
    assert (attempted["is_attack"] == 0).all()


def test_drop_policy_removes_attempted_rows(fixture_dir: Path) -> None:
    out, rep = clean(_df(fixture_dir), AttemptedPolicy.DROP)
    assert not out["label_raw"].astype(str).str.endswith(s.ATTEMPTED_SUFFIX).any()
    assert rep.rows_out == rep.rows_in - rep.attempted_rows - rep.duplicates_removed


def test_exact_duplicates_are_removed_and_counted_by_label(fixture_dir: Path) -> None:
    df = _df(fixture_dir)
    dup = df.iloc[[5, 5, 5]].copy()
    dup["id"] = [10_000_001, 10_000_002, 10_000_003]
    df2 = pd.concat([df, dup], ignore_index=True)
    _, rep_base = clean(df, AttemptedPolicy.RELABEL_BENIGN)
    _, rep = clean(df2, AttemptedPolicy.RELABEL_BENIGN)
    assert rep.duplicates_removed == rep_base.duplicates_removed + 3
    assert sum(rep.duplicates_removed_by_label.values()) == rep.duplicates_removed


def test_family_and_is_attack_are_consistent(fixture_dir: Path) -> None:
    out, rep = clean(_df(fixture_dir), AttemptedPolicy.RELABEL_BENIGN)
    assert set(out.loc[out["is_attack"] == 0, "family"].astype(str).unique()) == {"benign"}
    assert "benign" not in set(out.loc[out["is_attack"] == 1, "family"].astype(str).unique())
    assert out.loc[out["label"] == "Heartbleed", "family"].astype(str).eq("rare_exploit").all()
    assert sum(rep.family_counts_out.values()) == rep.rows_out


def test_report_round_trips_to_json(fixture_dir: Path, tmp_path: Path) -> None:
    _, rep = clean(_df(fixture_dir), AttemptedPolicy.RELABEL_BENIGN)
    p = rep.to_json(tmp_path / "r.json")
    loaded = json.loads(p.read_text())
    assert loaded["attempted_policy"] == "relabel_benign"
    assert loaded["rows_out"] == rep.rows_out


def test_report_counts_inf_cells_by_day(fixture_dir: Path) -> None:
    df = _df(fixture_dir)
    monday = df.index[df["day"] == "monday"][:2]
    friday = df.index[df["day"] == "friday"][:1]
    df.loc[monday, "Flow Bytes/s"] = np.float32("inf")
    df.loc[friday, "Flow Packets/s"] = np.float32("-inf")
    _, rep = clean(df, AttemptedPolicy.RELABEL_BENIGN)
    assert rep.inf_cells_replaced == 3
    assert rep.inf_cells_by_day == {"monday": 2, "friday": 1}
