from pathlib import Path

import mlflow
import pandas as pd
import pytest

from secops.config import Settings
from secops.data.build import build
from secops.data.clean import AttemptedPolicy
from secops.detection.features import FEATURE_SPEC_V1
from secops.detection.train import TrainConfig, run_training

pytestmark = pytest.mark.integration


@pytest.fixture
def built_settings(fixture_dir: Path, tmp_path: Path) -> Settings:
    st = Settings(data_dir=tmp_path, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    build(fixture_dir, st.processed_dir, st.reports_dir, AttemptedPolicy.RELABEL_BENIGN, subdir="")
    return st


def _artifact_paths(run_id: str) -> set[str]:
    return {a.path for a in mlflow.artifacts.list_artifacts(run_id=run_id)}


def test_binary_training_logs_metrics_and_artifacts(built_settings: Settings) -> None:
    cfg = TrainConfig(
        experiment="test/binary",
        run_name="lgbm",
        task="binary",
        model="lightgbm",
        weighting="balanced",
        params={"n_estimators": 30},
        shap_sample=50,
    )
    run_id = run_training(cfg, built_settings)
    mlflow.set_tracking_uri(built_settings.resolved_tracking_uri())
    run = mlflow.get_run(run_id)
    expected_metrics = (
        "val_pr_auc",
        "test_pr_auc",
        "test_recall",
        "test_fpr",
        "test_precision",
        "threshold",
    )
    for key in expected_metrics:
        assert key in run.data.metrics, key
    assert run.data.tags["attempted_policy"] == "relabel_benign"
    assert run.data.tags["feature_spec"] == "v1-noport"
    arts = _artifact_paths(run_id)
    for a in (
        "metrics.json",
        "threshold.json",
        "feature_spec.json",
        "pr_curve.png",
        "confusion_matrix.png",
        "per_label_recall.csv",
        "shap_global_importance.csv",
        "shap_summary.png",
        "explainer_background.parquet",
    ):
        assert a in arts, a
    # MLflow 3 stores the model as a LoggedModel, not a run artifact; the runs:/ URI must
    # still resolve and expose class probabilities (the contract Phase 2 depends on).
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    proba = loaded.predict(
        pd.read_parquet(built_settings.processed_dir / "relabel_benign" / "flows.parquet").pipe(
            lambda d: FEATURE_SPEC_V1.to_matrix(d.head(4))
        )
    )
    assert proba.shape == (4, 2)


def test_family_training_logs_per_class(built_settings: Settings) -> None:
    cfg = TrainConfig(
        experiment="test/family",
        run_name="lgbm_family",
        task="multiclass",
        model="lightgbm",
        weighting="balanced",
        params={"n_estimators": 30},
        shap_sample=50,
    )
    run_id = run_training(cfg, built_settings)
    mlflow.set_tracking_uri(built_settings.resolved_tracking_uri())
    run = mlflow.get_run(run_id)
    assert "test_macro_f1" in run.data.metrics
    assert any(k.startswith("test_recall_") for k in run.data.metrics)
    arts = _artifact_paths(run_id)
    assert "per_class_metrics.json" in arts and "confusion_matrix.png" in arts
    assert "threshold.json" not in arts


def test_logreg_training_runs(built_settings: Settings) -> None:
    cfg = TrainConfig(
        experiment="test/binary",
        run_name="lr",
        task="binary",
        model="logreg",
        weighting="none",
        shap_sample=50,
    )
    run_id = run_training(cfg, built_settings)
    mlflow.set_tracking_uri(built_settings.resolved_tracking_uri())
    assert "test_pr_auc" in mlflow.get_run(run_id).data.metrics
