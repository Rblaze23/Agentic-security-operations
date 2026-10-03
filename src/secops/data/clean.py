"""Cleaning: infinity handling, Attempted-flow policy, exact-duplicate removal, family labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from secops.data import schema as s


class AttemptedPolicy(StrEnum):
    RELABEL_BENIGN = "relabel_benign"  # dataset authors' recommendation (default)
    DROP = "drop"  # ablation


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    inf_cells_replaced: int
    attempted_rows: int
    attempted_policy: str
    duplicates_removed: int
    duplicates_removed_by_label: dict[str, int] = field(default_factory=dict)
    label_counts_in: dict[str, int] = field(default_factory=dict)
    label_counts_out: dict[str, int] = field(default_factory=dict)
    family_counts_out: dict[str, int] = field(default_factory=dict)

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def clean(df: pd.DataFrame, policy: AttemptedPolicy) -> tuple[pd.DataFrame, CleaningReport]:
    rows_in = len(df)
    out = df.copy()
    out["label_raw"] = out[s.LABEL_COL].astype(str)
    label_counts_in = _counts(out["label_raw"])

    attempted_mask = out["label_raw"].str.endswith(s.ATTEMPTED_SUFFIX)
    attempted_rows = int(attempted_mask.sum())
    if policy is AttemptedPolicy.DROP:
        out = out.loc[~attempted_mask].copy()
        out["label"] = out["label_raw"]
    else:
        out["label"] = np.where(attempted_mask, s.BENIGN_LABEL, out["label_raw"])

    feats = out[s.FEATURE_COLS].to_numpy(dtype=np.float32)
    inf_mask = np.isinf(feats)
    inf_cells = int(inf_mask.sum())
    if inf_cells:
        out[s.FEATURE_COLS] = np.where(inf_mask, np.nan, feats).astype(np.float32)

    dup_mask = out.duplicated(subset=[*s.FEATURE_COLS, "label"], keep="first")
    dup_by_label = _counts(out.loc[dup_mask, "label"])
    out = out.loc[~dup_mask].copy()

    out["family"] = out["label"].map(s.family_of).astype("category")
    out["is_attack"] = (out["label"] != s.BENIGN_LABEL).astype(np.int8)
    out["label"] = out["label"].astype("category")
    out["label_raw"] = out["label_raw"].astype("category")

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(out),
        inf_cells_replaced=inf_cells,
        attempted_rows=attempted_rows,
        attempted_policy=str(policy),
        duplicates_removed=int(dup_mask.sum()),
        duplicates_removed_by_label=dup_by_label,
        label_counts_in=label_counts_in,
        label_counts_out=_counts(out["label"]),
        family_counts_out=_counts(out["family"]),
    )
    return out.reset_index(drop=True), report
