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
