"""Training runs: load split Parquet, fit, evaluate, log to MLflow."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from secops.config import Settings
from secops.data import schema as s
from secops.data.clean import AttemptedPolicy
from secops.data.manifest import manifest_digest
from secops.data.split import SPLIT_COLUMN, SplitStrategy
from secops.detection.explain import global_importance, make_explainer
from secops.detection.features import FEATURE_SPECS, FeatureSpec
from secops.detection.metrics import (
    binary_metrics,
    max_f1_threshold,
    multiclass_metrics,
    per_group_recall,
    recall_at_fprs,
    select_threshold,
)
from secops.detection.models import (
    ModelName,
    Task,
    Weighting,
    build_model,
    fit_model,
    predict_proba_positive,
)
from secops.detection.plots import (
    confusion_matrix_figure,
    pr_curve_figure,
    save_figure,
    shap_bar_figure,
)

log = logging.getLogger(__name__)
BACKGROUND_ROWS = 1000


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str
    run_name: str
    task: Task
    model: ModelName
    weighting: Weighting = "balanced"
    params: dict[str, Any] = Field(default_factory=dict)
    feature_spec: str = "v1-noport"
    split_strategy: SplitStrategy = SplitStrategy.CHRONO_WITHIN_GROUP
    attempted_policy: AttemptedPolicy = AttemptedPolicy.RELABEL_BENIGN
    max_fpr: float = 0.01
    shap_sample: int = 20000
    reuse_threshold_from_run: str | None = None
    register_as: str | None = None
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
        data = yaml.safe_load(path.read_text()) or {}
        for k, v in (overrides or {}).items():
            data[k] = None if v == "" else v
        return cls.model_validate(data)


SWEEP_THRESHOLDS: tuple[float, ...] = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5)


def ensure_experiment(name: str, settings: Settings) -> str:
    """Create the experiment with its artifact root under the data directory, never <cwd>/mlruns."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return str(exp.experiment_id)
    location = settings.mlflow_dir / "artifacts" / name
    location.mkdir(parents=True, exist_ok=True)
    return str(mlflow.create_experiment(name, artifact_location=str(location)))


def load_parts(
    processed_dir: Path, policy: AttemptedPolicy, split_col: str
) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(processed_dir / str(policy) / "flows.parquet")
    return {name: df[df[split_col] == name] for name in ("train", "val", "test")}


def _git_sha() -> str:
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _select_rows(df: pd.DataFrame, task: Task) -> pd.DataFrame:
    if task == "binary":
        return df
    fam = df["family"].astype(str)
    return df[(df["is_attack"] == 1) & fam.isin(s.FAMILY_CLASSIFIER_CLASSES)]


def _targets(df: pd.DataFrame, task: Task) -> np.ndarray:
    if task == "binary":
        return df["is_attack"].to_numpy(dtype=int)
    return df["family"].astype(str).to_numpy()


def _log_model(model: Any, X_example: np.ndarray) -> None:
    """Log every estimator through the sklearn flavour with predict_proba as the pyfunc entry.

    The xgboost/lightgbm flavours' pyfunc wrappers return class labels, and the sklearn
    flavour's default skops format refuses to load numpy dtypes; cloudpickle + predict_proba
    gives one probability contract for Phase 2 regardless of the estimator.
    """
    sig = mlflow.models.infer_signature(X_example, model.predict_proba(X_example))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=sig,
        input_example=X_example[:5],
        pyfunc_predict_fn="predict_proba",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )


def _binary_stage(
    cfg: TrainConfig,
    model: Any,
    X: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    parts: dict[str, pd.DataFrame],
    out: Path,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    p_val = predict_proba_positive(model, X["val"])
    p_test = predict_proba_positive(model, X["test"])
    if cfg.reuse_threshold_from_run:
        prev = json.loads(
            mlflow.artifacts.load_text(f"runs:/{cfg.reuse_threshold_from_run}/threshold.json")
        )
        choice: dict[str, Any] = {
            "threshold": prev["threshold"],
            "method": f"reused:{cfg.reuse_threshold_from_run}",
        }
    else:
        c = select_threshold(y["val"], p_val, cfg.max_fpr)
        choice = {
            "threshold": c.threshold,
            "method": c.method,
            "val_fpr": c.fpr,
            "val_recall": c.recall,
            "max_f1_threshold": max_f1_threshold(y["val"], p_val).threshold,
        }
    t = float(choice["threshold"])
    for part, p in (("val", p_val), ("test", p_test)):
        bm = binary_metrics(y[part], p, t).to_dict()
        metrics.update({f"{part}_{k}": v for k, v in bm.items()})
        metrics.update({f"{part}_{k}": v for k, v in recall_at_fprs(y[part], p).items()})
    metrics["threshold"] = t
    (out / "threshold.json").write_text(json.dumps(choice, indent=2))
    labels = parts["test"]["label"].astype(str).to_numpy()
    per_label = per_group_recall(y["test"], p_test, t, labels)
    pd.Series(per_label, name="recall").rename_axis("label").to_csv(out / "per_label_recall.csv")
    sweep = {
        str(thr): {
            **binary_metrics(y["test"], p_test, thr).to_dict(),
            "per_label_recall": per_group_recall(y["test"], p_test, thr, labels),
        }
        for thr in (t, *SWEEP_THRESHOLDS)
    }
    (out / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))
    fp_mask = (y["test"] == 0) & (p_test >= t)
    fp_rows = parts["test"].loc[fp_mask]
    breakdown = {
        "threshold": t,
        "false_positives": int(fp_mask.sum()),
        "by_day": {str(k): int(v) for k, v in fp_rows["day"].astype(str).value_counts().items()},
        "by_label_raw": {
            str(k): int(v) for k, v in fp_rows["label_raw"].astype(str).value_counts().items()
        },
    }
    (out / "fp_breakdown.json").write_text(json.dumps(breakdown, indent=2))
    save_figure(pr_curve_figure(y["test"], p_test, t), out / "pr_curve.png")
    cm = [
        [int(metrics["test_tn"]), int(metrics["test_fp"])],
        [int(metrics["test_fn"]), int(metrics["test_tp"])],
    ]
    save_figure(confusion_matrix_figure(cm, ["benign", "attack"]), out / "confusion_matrix.png")
    return metrics


def _multiclass_stage(
    model: Any,
    X: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    classes: list[str],
    out: Path,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for part in ("val", "test"):
        proba = np.asarray(model.predict_proba(X[part]))
        pred = np.asarray(classes)[proba.argmax(axis=1)]
        mm = multiclass_metrics(y[part], pred, proba, classes)
        metrics[f"{part}_macro_f1"] = mm["macro_f1"]
        metrics[f"{part}_weighted_f1"] = mm["weighted_f1"]
        metrics[f"{part}_log_loss"] = mm["log_loss"]
        for c_name, d in mm["per_class"].items():
            metrics[f"{part}_recall_{c_name}"] = d["recall"]
            metrics[f"{part}_precision_{c_name}"] = d["precision"]
            metrics[f"{part}_support_{c_name}"] = d["support"]
        if part == "test":
            (out / "per_class_metrics.json").write_text(json.dumps(mm, indent=2))
            (out / "classes.json").write_text(json.dumps(list(classes)))
            fig = confusion_matrix_figure(mm["confusion"], classes)
            save_figure(fig, out / "confusion_matrix.png")
    return metrics


def _explain_stage(
    cfg: TrainConfig, model: Any, X_val: np.ndarray, spec: FeatureSpec, out: Path
) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.shap_sample, len(X_val))
    sample = X_val[rng.choice(len(X_val), n, replace=False)]
    background = sample[: min(BACKGROUND_ROWS, n)]
    explainer = make_explainer(model, cfg.model, background)
    imp = global_importance(explainer, sample, spec.names)
    imp.to_csv(out / "shap_global_importance.csv", index=False)
    save_figure(shap_bar_figure(imp), out / "shap_summary.png")
    pd.DataFrame(background, columns=spec.names).to_parquet(
        out / "explainer_background.parquet", index=False
    )


def run_training(cfg: TrainConfig, settings: Settings) -> str:
    mlflow.set_tracking_uri(settings.resolved_tracking_uri())
    mlflow.set_experiment(experiment_id=ensure_experiment(cfg.experiment, settings))
    spec = FEATURE_SPECS[cfg.feature_spec]
    parts = load_parts(
        settings.processed_dir, cfg.attempted_policy, SPLIT_COLUMN[cfg.split_strategy]
    )
    parts = {k: _select_rows(v, cfg.task) for k, v in parts.items()}
    X = {k: spec.to_matrix(v) for k, v in parts.items()}
    y = {k: _targets(v, cfg.task) for k, v in parts.items()}
    classes = list(s.FAMILY_CLASSIFIER_CLASSES) if cfg.task == "multiclass" else None
    if classes is not None:
        y_fit = {k: np.searchsorted(classes, v) for k, v in y.items()}
    else:
        y_fit = y

    with mlflow.start_run(run_name=cfg.run_name) as run, tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        mlflow.set_tags(
            {
                "dataset_variant": "distrinet_improved",
                "manifest_sha": manifest_digest(),
                "git_sha": _git_sha(),
                "split_strategy": str(cfg.split_strategy),
                "attempted_policy": str(cfg.attempted_policy),
                "feature_spec": cfg.feature_spec,
                "model": cfg.model,
                "weighting": cfg.weighting,
                "task": cfg.task,
            }
        )
        mlflow.log_params({"seed": cfg.seed, "max_fpr": cfg.max_fpr, "weighting": cfg.weighting})
        mlflow.log_params({f"p_{k}": v for k, v in cfg.params.items()})
        mlflow.log_params({f"rows_{k}": len(v) for k, v in parts.items()})

        model = build_model(
            cfg.model, cfg.task, cfg.params, cfg.seed, n_classes=len(classes) if classes else None
        )
        t0 = time.perf_counter()
        model = fit_model(
            model, X["train"], y_fit["train"], X["val"], y_fit["val"], cfg.weighting, cfg.model
        )
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        if classes is None:
            metrics = _binary_stage(cfg, model, X, y, parts, out)
        else:
            metrics = _multiclass_stage(model, X, y, classes, out)
        metrics["fit_seconds"] = fit_seconds
        metrics["eval_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _explain_stage(cfg, model, X["val"], spec, out)
        metrics["explain_seconds"] = time.perf_counter() - t0

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(float(v))})
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        spec.save(out / "feature_spec.json")

        mlflow.log_artifacts(str(out))
        _log_model(model, X["val"][:100])
        if cfg.register_as:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", cfg.register_as)
        headline = {
            k: round(float(v), 4)
            for k, v in metrics.items()
            if k.startswith("test_") and "support" not in k
        }
        log.info("run %s: %s", run.info.run_id, headline)
        return str(run.info.run_id)
