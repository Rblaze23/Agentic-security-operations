import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from secops.detection.features import FEATURE_SPEC_V1
from secops.detection.registry import CHAMPION_ALIAS, load_model, promote


@pytest.fixture
def tracking(tmp_path: Path) -> str:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    exp_id = mlflow.create_experiment(
        "test/registry", artifact_location=str(tmp_path / "artifacts")
    )
    mlflow.set_experiment(experiment_id=exp_id)
    return uri


def _fake_run(workdir: Path) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    X = np.random.default_rng(0).normal(size=(50, 82)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    with mlflow.start_run() as run:
        FEATURE_SPEC_V1.save(workdir / "feature_spec.json")
        (workdir / "threshold.json").write_text(json.dumps({"threshold": 0.37, "method": "test"}))
        mlflow.log_artifact(str(workdir / "feature_spec.json"))
        mlflow.log_artifact(str(workdir / "threshold.json"))
        mlflow.sklearn.log_model(
            clf,
            name="model",
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        return str(run.info.run_id)


def test_promote_then_load_round_trip(tracking: str, tmp_path: Path) -> None:
    run_id = _fake_run(tmp_path / "a")
    assert promote(run_id, "test-detector", CHAMPION_ALIAS) == 1
    loaded = load_model("test-detector", CHAMPION_ALIAS)
    assert loaded.run_id == run_id
    assert loaded.threshold == pytest.approx(0.37)
    assert loaded.feature_spec == FEATURE_SPEC_V1
    assert loaded.classes is None
    proba = loaded.model.predict_proba(np.zeros((2, 82), dtype=np.float32))
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_promote_second_run_moves_alias(tracking: str, tmp_path: Path) -> None:
    r1 = _fake_run(tmp_path / "a")
    r2 = _fake_run(tmp_path / "b")
    promote(r1, "test-detector-2")
    assert promote(r2, "test-detector-2") == 2
    assert load_model("test-detector-2").run_id == r2


def test_load_model_raises_on_artifact_store_error(
    tracking: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _fake_run(tmp_path / "c")
    promote(run_id, "test-detector-3")
    import secops.detection.registry as reg

    def boom(*_a: object, **_k: object) -> str:
        raise OSError("artifact store unreachable")

    monkeypatch.setattr(reg.mlflow.artifacts, "load_text", boom)
    with pytest.raises(OSError):
        load_model("test-detector-3")


def test_load_model_reads_classes_json(tracking: str, tmp_path: Path) -> None:
    workdir = tmp_path / "d"
    workdir.mkdir()
    X = np.random.default_rng(0).normal(size=(60, 82)).astype(np.float32)
    y = np.repeat([0, 1, 2], 20)
    clf = LogisticRegression().fit(X, y)
    with mlflow.start_run() as run:
        FEATURE_SPEC_V1.save(workdir / "feature_spec.json")
        (workdir / "classes.json").write_text(json.dumps(["zeta", "alpha", "mid"]))
        mlflow.log_artifact(str(workdir / "feature_spec.json"))
        mlflow.log_artifact(str(workdir / "classes.json"))
        mlflow.sklearn.log_model(
            clf,
            name="model",
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    promote(str(run.info.run_id), "test-family")
    loaded = load_model("test-family")
    assert loaded.classes == ["zeta", "alpha", "mid"]  # training order, not sorted
    assert loaded.threshold is None
