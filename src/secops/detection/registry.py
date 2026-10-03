"""MLflow Model Registry helpers: promote a run to an alias; load a model with its spec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
from mlflow import MlflowClient

from secops.detection.features import FeatureSpec

DETECTOR_MODEL_NAME = "secops-detector"
FAMILY_MODEL_NAME = "secops-family-classifier"
CHAMPION_ALIAS = "champion"


@dataclass
class LoadedModel:
    model: Any
    feature_spec: FeatureSpec
    threshold: float | None
    classes: list[str] | None
    model_name: str
    version: int
    run_id: str


class _ProbaAdapter:
    """Normalise pyfunc output to an (n, n_classes) probability matrix."""

    def __init__(self, pyfunc_model: Any) -> None:
        self._m = pyfunc_model

    def predict_proba(self, X: Any) -> np.ndarray:
        out = np.asarray(self._m.predict(X), dtype=float)
        if out.ndim == 1:  # positive-class probability only
            out = np.column_stack([1 - out, out])
        return out


def promote(run_id: str, model_name: str, alias: str = CHAMPION_ALIAS) -> int:
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.set_registered_model_alias(model_name, alias, mv.version)
    return int(mv.version)


def _artifact_json(run_id: str, name: str) -> dict[str, Any] | None:
    try:
        return dict(json.loads(mlflow.artifacts.load_text(f"runs:/{run_id}/{name}")))
    except Exception:  # noqa: BLE001  artifact absent for this task type
        return None


def load_model(model_name: str, alias: str = CHAMPION_ALIAS) -> LoadedModel:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    run_id = str(mv.run_id)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
    spec_json = _artifact_json(run_id, "feature_spec.json")
    if spec_json is None:
        raise ValueError(f"run {run_id} has no feature_spec.json artifact")
    thr = _artifact_json(run_id, "threshold.json")
    per_class = _artifact_json(run_id, "per_class_metrics.json")
    return LoadedModel(
        model=_ProbaAdapter(pyfunc),
        feature_spec=FeatureSpec.model_validate(spec_json),
        threshold=float(thr["threshold"]) if thr else None,
        classes=sorted(per_class["per_class"]) if per_class else None,
        model_name=model_name,
        version=int(mv.version),
        run_id=run_id,
    )
