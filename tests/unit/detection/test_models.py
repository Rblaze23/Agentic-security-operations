import numpy as np
import pytest

from secops.detection.models import build_model, fit_model, predict_proba_positive

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 10)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
X[::50, 2] = np.nan  # NaN cells like inf->nan rate features
Xv, yv = X[:100], y[:100]


@pytest.mark.parametrize("name", ["logreg", "xgboost", "lightgbm"])
@pytest.mark.parametrize("weighting", ["none", "balanced"])
def test_binary_models_fit_and_score(name: str, weighting: str) -> None:
    m = build_model(name, "binary", {}, seed=1)  # type: ignore[arg-type]
    m = fit_model(m, X, y, Xv, yv, weighting=weighting, name=name)  # type: ignore[arg-type]
    p = predict_proba_positive(m, Xv)
    assert p.shape == (100,) and (0 <= p).all() and (p <= 1).all()
    assert ((p >= 0.5).astype(int) == yv).mean() > 0.8


def test_logreg_pipeline_handles_nan() -> None:
    m = build_model("logreg", "binary", {}, seed=1)
    m = fit_model(m, X, y, Xv, yv, weighting="none", name="logreg")
    assert np.isfinite(predict_proba_positive(m, Xv)).all()


def test_logreg_multiclass_not_supported() -> None:
    with pytest.raises(ValueError):
        build_model("logreg", "multiclass", {}, seed=1, n_classes=3)


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_multiclass_models(name: str) -> None:
    ym = np.digitize(X[:, 0], [-0.5, 0.5])  # classes 0, 1, 2
    m = build_model(name, "multiclass", {}, seed=1, n_classes=3)  # type: ignore[arg-type]
    m = fit_model(m, X, ym, Xv, ym[:100], weighting="balanced", name=name)  # type: ignore[arg-type]
    proba = m.predict_proba(Xv)
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
