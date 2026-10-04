"""SHAP explanations for the detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from secops.detection.models import ModelName


@dataclass
class FeatureContribution:
    feature: str
    value: float
    shap_value: float


class _PipelineExplainer:
    """LinearExplainer over the fitted imputer+scaler of the logistic-regression pipeline."""

    def __init__(self, transform: Any, inner: Any) -> None:
        self.transform, self.inner = transform, inner

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.shap_values(self.transform.transform(X)))


def make_explainer(model: Any, name: ModelName, X_background: np.ndarray) -> Any:
    if name in ("xgboost", "lightgbm"):
        return shap.TreeExplainer(model)
    transform = model[:-1]
    clf = model[-1]
    return _PipelineExplainer(
        transform, shap.LinearExplainer(clf, transform.transform(X_background))
    )


def _shap_tensor(explainer: Any, X: np.ndarray) -> np.ndarray:
    """SHAP values as (n, features) for binary models or (n, features, classes) for multiclass."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        if len(sv) == 2:  # older shap API for binary tree models: [neg, pos]
            return np.asarray(sv[-1])
        return np.stack([np.asarray(a) for a in sv], axis=-1)
    arr = np.asarray(sv)
    if arr.ndim == 3 and arr.shape[-1] == 2:  # (n, features, 2): binary, keep positive class
        return np.asarray(arr[:, :, -1])
    return arr


def global_importance(explainer: Any, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature; for multiclass models averaged over classes as well."""
    sv = _shap_tensor(explainer, X)
    axes = (0, 2) if sv.ndim == 3 else (0,)
    imp = np.abs(sv).mean(axis=axes)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_k_contributions(
    explainer: Any,
    x: np.ndarray,
    feature_names: list[str],
    k: int = 5,
    class_index: int | None = None,
) -> list[FeatureContribution]:
    """Top-k contributions for one row: positive class for binary, `class_index` for multiclass."""
    sv = _shap_tensor(explainer, x.reshape(1, -1))[0]
    if sv.ndim == 2:
        if class_index is None:
            raise ValueError("class_index is required for a multiclass explainer")
        sv = sv[:, class_index]
    order = np.argsort(-np.abs(sv))[:k]
    return [FeatureContribution(feature_names[i], float(x[i]), float(sv[i])) for i in order]


def top_k_contributions_batch(
    explainer: Any,
    X: np.ndarray,
    feature_names: list[str],
    k: int = 5,
    class_index: int | None = None,
) -> list[list[FeatureContribution]]:
    """`top_k_contributions` for every row of X with a single SHAP call."""
    sv = _shap_tensor(explainer, X)
    if sv.ndim == 3:
        if class_index is None:
            raise ValueError("class_index is required for a multiclass explainer")
        sv = sv[:, :, class_index]
    out: list[list[FeatureContribution]] = []
    for i in range(X.shape[0]):
        order = np.argsort(-np.abs(sv[i]))[:k]
        out.append(
            [FeatureContribution(feature_names[j], float(X[i, j]), float(sv[i, j])) for j in order]
        )
    return out
