from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from secops.config import Settings
from secops.data.build import build
from secops.data.clean import AttemptedPolicy
from secops.detection.bundle import export_bundle, load_bundle
from secops.detection.registry import promote
from secops.detection.train import TrainConfig, run_training

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def trained(fixture_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    tmp = tmp_path_factory.mktemp("bundle")
    st = Settings(data_dir=tmp, mlflow_tracking_uri=f"sqlite:///{tmp / 'mlflow.db'}")
    build(fixture_dir, st.processed_dir, st.reports_dir, AttemptedPolicy.RELABEL_BENIGN, subdir="")
    det = run_training(
        TrainConfig(
            experiment="t/b",
            run_name="det",
            task="binary",
            model="lightgbm",
            weighting="none",
            params={"n_estimators": 20},
            shap_sample=40,
        ),
        st,
    )
    fam = run_training(
        TrainConfig(
            experiment="t/f",
            run_name="fam",
            task="multiclass",
            model="lightgbm",
            weighting="none",
            params={"n_estimators": 20},
            shap_sample=40,
        ),
        st,
    )
    mlflow.set_tracking_uri(st.resolved_tracking_uri())
    promote(det, "t-detector")
    promote(fam, "t-family")
    return st


def test_export_then_load_matches_registry_predictions(trained: Settings, tmp_path: Path) -> None:
    mlflow.set_tracking_uri(trained.resolved_tracking_uri())
    manifest = export_bundle("t-detector", tmp_path / "detector")
    assert manifest.model_name == "t-detector" and manifest.version == 1
    assert manifest.model_kind == "lightgbm" and manifest.feature_spec_version == "v1-noport"
    assert "test_pr_auc" in manifest.metrics and "manifest_sha" in manifest.tags
    for f in (
        "model/MLmodel",
        "feature_spec.json",
        "threshold.json",
        "explainer_background.parquet",
        "bundle.json",
    ):
        assert (tmp_path / "detector" / f).exists(), f
    b = load_bundle(tmp_path / "detector")
    assert b.threshold is not None and b.background is not None and b.background.shape[1] == 82
    df = pd.read_parquet(trained.processed_dir / "relabel_benign" / "flows.parquet").head(5)
    X = b.feature_spec.to_matrix(df)
    from_bundle = b.estimator.predict_proba(X)[:, 1]
    from_registry = np.asarray(mlflow.pyfunc.load_model("models:/t-detector@champion").predict(X))[
        :, 1
    ]
    assert np.allclose(from_bundle, from_registry)


def test_export_family_bundle_has_classes(trained: Settings, tmp_path: Path) -> None:
    mlflow.set_tracking_uri(trained.resolved_tracking_uri())
    export_bundle("t-family", tmp_path / "family")
    b = load_bundle(tmp_path / "family")
    assert b.classes == ["botnet", "brute_force", "ddos", "dos", "port_scan", "web_attack"]
    assert b.threshold is None


def test_export_overwrites_existing_directory(trained: Settings, tmp_path: Path) -> None:
    mlflow.set_tracking_uri(trained.resolved_tracking_uri())
    out = tmp_path / "again"
    export_bundle("t-detector", out)
    (out / "stale.txt").write_text("old")
    export_bundle("t-detector", out)
    assert not (out / "stale.txt").exists() and (out / "bundle.json").exists()


def test_exported_bundle_is_readable_by_other_users(trained: Settings, tmp_path: Path) -> None:
    """The container runs as a non-root user; a 0700 bundle (mkdtemp default) is unreadable."""
    mlflow.set_tracking_uri(trained.resolved_tracking_uri())
    out = tmp_path / "perm"
    export_bundle("t-detector", out)
    assert out.stat().st_mode & 0o055 == 0o055, oct(out.stat().st_mode)
    assert (out / "model").stat().st_mode & 0o055 == 0o055
    assert (out / "bundle.json").stat().st_mode & 0o044 == 0o044


def test_export_failure_keeps_previous_bundle(
    trained: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow.set_tracking_uri(trained.resolved_tracking_uri())
    out = tmp_path / "keep"
    export_bundle("t-detector", out)
    marker = out / "bundle.json"
    before = marker.read_text()

    import secops.detection.bundle as bundle_mod

    real_rename = Path.rename

    def failing_rename(self: Path, target: Path) -> Path:
        if Path(target) == out and self != out and not str(self).endswith(".old"):
            raise OSError("simulated cross-device rename failure")
        return real_rename(self, target)

    monkeypatch.setattr(bundle_mod.Path, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated"):
        export_bundle("t-detector", out)
    assert marker.read_text() == before, "previous bundle must survive a failed swap"
    assert not out.with_name(out.name + ".old").exists()
