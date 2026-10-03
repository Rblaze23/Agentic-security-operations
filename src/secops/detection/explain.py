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


def _positive_class_shap(explainer: Any, X: np.ndarray) -> np.ndarray:
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older shap API for binary tree models: [neg, pos]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        sv = sv[:, :, -1]
    return np.asarray(sv)


def global_importance(explainer: Any, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    sv = _positive_class_shap(explainer, X)
    imp = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_k_contributions(
    explainer: Any, x: np.ndarray, feature_names: list[str], k: int = 5
) -> list[FeatureContribution]:
    sv = _positive_class_shap(explainer, x.reshape(1, -1))[0]
    order = np.argsort(-np.abs(sv))[:k]
    return [FeatureContribution(feature_names[i], float(x[i]), float(sv[i])) for i in order]
