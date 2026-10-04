"""DetectorService: bundles in, typed predictions out. No HTTP, no MLflow."""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from secops.detection.bundle import BundleError, LoadedBundle, load_bundle
from secops.detection.explain import make_explainer, top_k_contributions_batch
from secops.schemas.api import BundleInfo, ModelInfo
from secops.schemas.flow import FeatureMismatchError, PredictRequest, validate_features
from secops.schemas.prediction import FeatureContribution, Prediction


def _bundle_info(b: LoadedBundle) -> BundleInfo:
    m = b.manifest
    return BundleInfo(
        model_name=m.model_name,
        version=m.version,
        run_id=m.run_id,
        alias=m.alias,
        exported_at=datetime.fromisoformat(m.exported_at),
        feature_spec_version=m.feature_spec_version,
        model_kind=m.model_kind,
        metrics=m.metrics,
        tags=m.tags,
    )


class DetectorService:
    """Binary detector + optional family classifier + SHAP explainer, loaded once."""

    def __init__(self, detector: LoadedBundle, family: LoadedBundle | None, k: int = 5) -> None:
        if detector.threshold is None:
            raise BundleError(f"{detector.path}: detector bundle has no threshold.json")
        if family is not None:
            if family.classes is None:
                raise BundleError(f"{family.path}: family bundle has no classes.json")
            if family.feature_spec != detector.feature_spec:
                raise BundleError(
                    "family bundle feature spec "
                    f"({family.feature_spec.version}) differs from the detector's "
                    f"({detector.feature_spec.version})"
                )
        self.detector = detector
        self.family = family
        self.k = k
        self.threshold: float = detector.threshold
        self.spec = detector.feature_spec
        kind: Any = detector.manifest.model_kind
        if kind == "logreg" and detector.background is None:
            raise BundleError(
                f"{detector.path}: logistic-regression bundle needs a background sample"
            )
        background = (
            detector.background
            if detector.background is not None
            else np.zeros((1, len(self.spec.names)), dtype=np.float32)
        )
        self.explainer = make_explainer(detector.estimator, kind, background)

    @classmethod
    def from_dirs(
        cls, detector_dir: Path, family_dir: Path | None = None, k: int = 5
    ) -> DetectorService:
        det = load_bundle(Path(detector_dir))
        fam = load_bundle(Path(family_dir)) if family_dir is not None else None
        return cls(det, fam, k=k)

    @property
    def ready(self) -> bool:
        return True

    def info(self) -> ModelInfo:
        return ModelInfo(
            detector=_bundle_info(self.detector),
            family=_bundle_info(self.family) if self.family is not None else None,
            feature_spec_version=self.spec.version,
            threshold=self.threshold,
            top_k=self.k,
        )

    def predict(self, requests: list[PredictRequest]) -> list[Prediction]:
        """Score flows. Raises ValueError on feature-name drift (the API maps it to 422)."""
        start = time.perf_counter()
        rows = []
        for i, r in enumerate(requests):
            try:
                rows.append(validate_features(r.features, self.spec))
            except FeatureMismatchError as e:
                raise FeatureMismatchError(f"item {i}: {e}") from e
        frame = pd.DataFrame(rows, columns=self.spec.names)
        X = self.spec.to_matrix(frame)
        proba = np.asarray(self.detector.estimator.predict_proba(X))[:, 1]
        is_alert = proba >= self.threshold

        families: dict[int, tuple[str, dict[str, float]]] = {}
        if self.family is not None and is_alert.any():
            idx = np.flatnonzero(is_alert)
            fam_proba = np.asarray(self.family.estimator.predict_proba(X[idx]))
            classes = self.family.classes or []
            for row_i, probs in zip(idx, fam_proba, strict=True):
                dist = {c: float(v) for c, v in zip(classes, probs, strict=True)}
                families[int(row_i)] = (classes[int(np.argmax(probs))], dist)

        contributions = top_k_contributions_batch(self.explainer, X, self.spec.names, k=self.k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        m = self.detector.manifest
        out: list[Prediction] = []
        for i, req in enumerate(requests):
            fam = families.get(i)
            out.append(
                Prediction(
                    event_id=req.event_id,
                    model_name=m.model_name,
                    model_version=m.version,
                    run_id=m.run_id,
                    feature_spec_version=self.spec.version,
                    attack_probability=float(min(max(proba[i], 0.0), 1.0)),
                    threshold=self.threshold,
                    is_alert=bool(is_alert[i]),
                    predicted_family=fam[0] if fam else None,
                    family_probabilities=fam[1] if fam else None,
                    top_contributions=[
                        FeatureContribution(
                            feature=c.feature,
                            value=None if math.isnan(c.value) else c.value,
                            shap_value=c.shap_value,
                        )
                        for c in contributions[i]
                    ],
                    latency_ms=latency_ms,
                )
            )
        return out
