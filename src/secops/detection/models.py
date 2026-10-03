"""Model factory and fitting helpers for the three baselines."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ModelName = Literal["logreg", "xgboost", "lightgbm"]
Task = Literal["binary", "multiclass"]
Weighting = Literal["none", "balanced"]

EARLY_STOPPING_ROUNDS = 50

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
    "xgboost": {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    },
    "lightgbm": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    },
}


def build_model(
    name: ModelName,
    task: Task,
    params: dict[str, Any],
    seed: int,
    n_classes: int | None = None,
) -> Any:
    p = {**DEFAULT_PARAMS[name], **params}
    if name == "logreg":
        if task == "multiclass":
            raise ValueError("logreg is a binary-only baseline in Phase 1")
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(random_state=seed, **p)),
            ]
        )
    if name == "xgboost":
        if task == "binary":
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=seed,
                n_jobs=-1,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                **p,
            )
        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **p,
        )
    if name == "lightgbm":
        objective = "binary" if task == "binary" else "multiclass"
        return LGBMClassifier(objective=objective, random_state=seed, n_jobs=-1, verbose=-1, **p)
    raise ValueError(name)


def fit_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weighting: Weighting,
    name: ModelName,
) -> Any:
    sw = compute_sample_weight("balanced", y_train) if weighting == "balanced" else None
    if name == "logreg":
        model.fit(X_train, y_train, clf__sample_weight=sw)
    elif name == "xgboost":
        model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "lightgbm":
        metric = "average_precision" if len(np.unique(y_train)) == 2 else "multi_logloss"
        model.fit(
            X_train,
            y_train,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            eval_metric=metric,
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
        )
    else:
        raise ValueError(name)
    return model


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]
