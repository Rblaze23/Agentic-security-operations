from typing import Any

import numpy as np

from secops.detection.explain import global_importance, make_explainer, top_k_contributions
from secops.detection.models import build_model, fit_model

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 6)).astype(np.float32)
y = (2 * X[:, 0] - X[:, 3] > 0).astype(int)
names = [f"f{i}" for i in range(6)]


def _fit(name: Any) -> Any:
    params = {"n_estimators": 50} if name != "logreg" else {}
    m = build_model(name, "binary", params, seed=0)
    return fit_model(m, X, y, X[:50], y[:50], weighting="none", name=name)


def test_global_importance_ranks_true_drivers_first() -> None:
    for name in ("lightgbm", "xgboost", "logreg"):
        ex = make_explainer(_fit(name), name, X[:100])  # type: ignore[arg-type]
        imp = global_importance(ex, X[:100], names)
        assert list(imp.columns) == ["feature", "mean_abs_shap"]
        assert set(imp["feature"].head(2)) == {"f0", "f3"}, name


def test_top_k_contributions_sorted_by_magnitude() -> None:
    ex = make_explainer(_fit("lightgbm"), "lightgbm", X[:100])
    top = top_k_contributions(ex, X[0], names, k=3)
    assert len(top) == 3
    mags = [abs(c.shap_value) for c in top]
    assert mags == sorted(mags, reverse=True)
    assert top[0].feature in {"f0", "f3"}


def test_multiclass_global_importance_averages_over_classes() -> None:
    import shap

    ym = np.digitize(X[:, 0], [-0.5, 0.5])
    m = build_model("lightgbm", "multiclass", {"n_estimators": 30}, seed=0, n_classes=3)
    m = fit_model(m, X, ym, X[:50], ym[:50], weighting="none", name="lightgbm")
    ex = make_explainer(m, "lightgbm", X[:100])
    imp = global_importance(ex, X[:100], names)
    raw = np.asarray(shap.TreeExplainer(m).shap_values(X[:100]))
    if raw.ndim == 3 and raw.shape[0] == 3:  # (classes, n, features) layout
        raw = np.moveaxis(raw, 0, -1)
    expected = np.abs(raw).mean(axis=(0, 2))  # mean over rows and classes
    got = imp.set_index("feature").loc[names, "mean_abs_shap"].to_numpy()
    assert np.allclose(got, expected)
