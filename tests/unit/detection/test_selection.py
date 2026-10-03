import pandas as pd

from secops.detection.selection import choose_champion


def _runs(rows: list[tuple[str, str, str, float, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": rid,
                "tags.mlflow.runName": name,
                "tags.model": model,
                "metrics.val_pr_auc": metric,
                "metrics.val_fpr": fpr,
                "status": status,
            }
            for rid, name, model, metric, fpr, status in rows
        ]
    )


def test_clear_winner_takes_the_metric() -> None:
    df = _runs(
        [
            ("a", "lr", "logreg", 0.90, 0.01, "FINISHED"),
            ("b", "xg", "xgboost", 0.95, 0.01, "FINISHED"),
        ]
    )
    assert choose_champion(df, "val_pr_auc").run_id == "b"


def test_tie_band_prefers_simpler_model_then_lower_val_fpr() -> None:
    df = _runs(
        [
            ("xgb", "xgboost_balanced", "xgboost", 0.999995, 0.0047, "FINISHED"),
            ("lgb_b", "lightgbm_balanced", "lightgbm", 0.999993, 0.0077, "FINISHED"),
            ("lgb_n", "lightgbm_none", "lightgbm", 0.999993, 0.0043, "FINISHED"),
            ("lr", "logreg", "logreg", 0.9947, 0.0096, "FINISHED"),
        ]
    )
    choice = choose_champion(df, "val_pr_auc", tie_band=0.0005)
    assert choice.run_id == "lgb_n"
    assert choice.reason.startswith("tie")


def test_unfinished_runs_are_ignored() -> None:
    df = _runs(
        [
            ("zombie", "z", "xgboost", 1.0, 0.0, "RUNNING"),
            ("ok", "o", "lightgbm", 0.9, 0.01, "FINISHED"),
        ]
    )
    assert choose_champion(df, "val_pr_auc").run_id == "ok"


def test_metric_without_fpr_column_still_works() -> None:
    df = _runs([("a", "x", "xgboost", 0.99, 0.0, "FINISHED")]).drop(columns=["metrics.val_fpr"])
    assert choose_champion(df, "val_pr_auc").run_id == "a"
