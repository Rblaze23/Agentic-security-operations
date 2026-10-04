"""secops-train command line."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import mlflow
import pandas as pd
import typer

from secops.config import get_settings
from secops.detection.bundle import export_bundle
from secops.detection.registry import promote
from secops.detection.selection import choose_champion
from secops.detection.train import TrainConfig, run_training

app = typer.Typer(help="Train and report detection models.")

REPORT_COLS = [
    "run_name",
    "model",
    "weighting",
    "val_pr_auc",
    "test_pr_auc",
    "test_recall",
    "test_fpr",
    "test_precision",
    "test_f1",
    "test_roc_auc",
    "threshold",
    "val_macro_f1",
    "test_macro_f1",
]


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"expected key=value, got {item!r}")
        out[k] = v
    return out


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", help="YAML training config")],
    set_: Annotated[
        list[str], typer.Option("--set", help="key=value overrides of the YAML config")
    ] = [],  # noqa: B006
) -> None:
    """Train one configuration and print the MLflow run id."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainConfig.from_yaml(config, _parse_overrides(set_))
    typer.echo(run_training(cfg, get_settings()))


@app.command()
def report(
    experiment: Annotated[str, typer.Option("--experiment", help="MLflow experiment name")],
) -> None:
    """Print a markdown table of all runs in an experiment (numbers come only from MLflow)."""
    mlflow.set_tracking_uri(get_settings().resolved_tracking_uri())
    df = mlflow.search_runs(experiment_names=[experiment])
    if df.empty:
        typer.echo("no runs")
        raise typer.Exit(code=1)
    rows = []
    for _, r in df.iterrows():
        row: dict[str, Any] = {
            "run_name": r.get("tags.mlflow.runName"),
            "model": r.get("tags.model"),
            "weighting": r.get("tags.weighting"),
            "run_id": str(r["run_id"])[:8],
        }
        for c in REPORT_COLS[3:]:
            v = r.get(f"metrics.{c}")
            row[c] = f"{v:.4f}" if isinstance(v, float) and pd.notna(v) else ""
        rows.append(row)
    typer.echo(pd.DataFrame(rows, columns=[*REPORT_COLS, "run_id"]).to_markdown(index=False))


@app.command(name="promote-best")
def promote_best(
    experiment: Annotated[str, typer.Option("--experiment", help="MLflow experiment name")],
    model_name: Annotated[str, typer.Option("--model-name", help="registered model name")],
    metric: Annotated[str, typer.Option("--metric")] = "val_pr_auc",
    tie_band: Annotated[float, typer.Option("--tie-band")] = 0.0005,
) -> None:
    """Promote the champion by the documented rule and point the champion alias at it."""
    mlflow.set_tracking_uri(get_settings().resolved_tracking_uri())
    df = mlflow.search_runs(experiment_names=[experiment])
    if df.empty or f"metrics.{metric}" not in df:
        typer.echo(f"no runs with {metric} in {experiment}")
        raise typer.Exit(code=1)
    choice = choose_champion(df, metric, tie_band=tie_band)
    version = promote(choice.run_id, model_name)
    typer.echo(
        f"{model_name} v{version} <- {choice.run_name} ({choice.run_id}) "
        f"{metric}={choice.metric_value:.6f}; {choice.reason}"
    )


@app.command()
def export(
    model_name: Annotated[str, typer.Option("--model-name", help="registered model name")],
    out: Annotated[Path, typer.Option("--out", help="bundle directory to (re)create")],
    alias: Annotated[str, typer.Option("--alias")] = "champion",
) -> None:
    """Write a self-contained serving bundle for a registry alias."""
    mlflow.set_tracking_uri(get_settings().resolved_tracking_uri())
    m = export_bundle(model_name, out, alias=alias)
    typer.echo(
        f"exported {m.model_name} v{m.version} ({m.alias}, run {m.run_id[:8]}, "
        f"{m.model_kind}, spec {m.feature_spec_version}) -> {out}"
    )
