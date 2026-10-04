import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from secops.detection.bundle import BundleError, BundleManifest, load_bundle
from secops.detection.features import FEATURE_SPEC_V1, FeatureSpec


def _manifest(**over: object) -> BundleManifest:
    base: dict[str, object] = {
        "model_name": "secops-detector",
        "version": 1,
        "run_id": "abc",
        "alias": "champion",
        "exported_at": datetime(2026, 10, 3, tzinfo=UTC).isoformat(),
        "feature_spec_version": "v1-noport",
        "model_kind": "logreg",
        "metrics": {"test_pr_auc": 0.5},
        "tags": {"git_sha": "x"},
    }
    base.update(over)
    return BundleManifest(**base)  # type: ignore[arg-type]


def _write_bundle(root: Path, n_features: int, classes: list[str] | None = None) -> Path:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, n_features)).astype(np.float32)
    y = (
        (X[:, 0] > 0).astype(int)
        if classes is None
        else np.repeat(range(len(classes)), 60 // len(classes))[:60]
    )
    clf = LogisticRegression(max_iter=200).fit(X, y)
    root.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(clf, str(root / "model"))
    FEATURE_SPEC_V1.save(root / "feature_spec.json")
    if classes is None:
        (root / "threshold.json").write_text(json.dumps({"threshold": 0.4, "method": "test"}))
    else:
        (root / "classes.json").write_text(json.dumps(classes))
    _manifest(model_kind="logreg").to_json(root / "bundle.json")
    return root


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = _manifest()
    p = m.to_json(tmp_path / "bundle.json")
    assert BundleManifest.from_json(p) == m


def test_load_detector_bundle(tmp_path: Path) -> None:
    b = load_bundle(_write_bundle(tmp_path / "det", 82))
    assert b.threshold == pytest.approx(0.4) and b.classes is None
    assert b.feature_spec == FEATURE_SPEC_V1 and b.manifest.model_kind == "logreg"
    proba = b.estimator.predict_proba(np.zeros((3, 82), dtype=np.float32))
    assert proba.shape == (3, 2)


def test_load_family_bundle_reads_classes_in_order(tmp_path: Path) -> None:
    b = load_bundle(_write_bundle(tmp_path / "fam", 82, classes=["zeta", "alpha", "mid"]))
    assert b.classes == ["zeta", "alpha", "mid"] and b.threshold is None


def test_bundle_rejects_mismatched_feature_spec(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bad", 82)
    FeatureSpec(names=FEATURE_SPEC_V1.names[:-1], version="broken").save(root / "feature_spec.json")
    with pytest.raises(BundleError, match="81"):
        load_bundle(root)


def test_bundle_rejects_class_count_mismatch(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bad2", 82, classes=["a", "b", "c"])
    (root / "classes.json").write_text(json.dumps(["a", "b"]))
    with pytest.raises(BundleError, match="classes"):
        load_bundle(root)


def test_bundle_requires_feature_spec(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "nospec", 82)
    (root / "feature_spec.json").unlink()
    with pytest.raises(BundleError, match="feature_spec.json"):
        load_bundle(root)
