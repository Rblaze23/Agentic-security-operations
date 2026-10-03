"""Champion selection: best validation metric, ties resolved toward the simpler model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MODEL_SIMPLICITY: dict[str, int] = {"logreg": 0, "lightgbm": 1, "xgboost": 2}


@dataclass
class ChampionChoice:
    run_id: str
    run_name: str
    metric_value: float
    reason: str


def choose_champion(
    runs: pd.DataFrame,
    metric: str,
    tie_band: float = 0.0005,
    fpr_metric: str = "val_fpr",
) -> ChampionChoice:
    """Pick a run from an ``mlflow.search_runs`` frame.

    Rule (docs/evaluation.md): highest ``metrics.<metric>`` among FINISHED runs; every run within
    ``tie_band`` of the best is a tie, resolved by the simpler model (logreg < lightgbm < xgboost),
    then by the lower ``metrics.<fpr_metric>`` when that column exists, then by the metric itself.
    """
    col = f"metrics.{metric}"
    df = runs[(runs["status"] == "FINISHED") & runs[col].notna()].copy()
    if df.empty:
        raise ValueError(f"no finished runs with {metric}")
    best = float(df[col].max())
    tied = df[df[col] >= best - tie_band].copy()
    if len(tied) == 1:
        row = tied.iloc[0]
        reason = f"best {metric}"
    else:
        tied["_simplicity"] = tied["tags.model"].map(MODEL_SIMPLICITY).fillna(99)
        fpr_col = f"metrics.{fpr_metric}"
        if fpr_col in tied:
            tied["_fpr"] = tied[fpr_col].fillna(float("inf"))
        else:
            tied["_fpr"] = 0.0
        tied["_neg_metric"] = -tied[col]
        row = tied.sort_values(["_simplicity", "_fpr", "_neg_metric"]).iloc[0]
        reason = (
            f"tie within {tie_band} of best {metric} ({len(tied)} runs): "
            f"simpler model, then lower {fpr_metric}"
        )
    return ChampionChoice(
        run_id=str(row["run_id"]),
        run_name=str(row.get("tags.mlflow.runName", "")),
        metric_value=float(row[col]),
        reason=reason,
    )
