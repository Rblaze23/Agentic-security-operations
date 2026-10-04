"""Build small serving bundles from the mini_cicids fixture without an MLflow server.

Used by the API unit tests and by the CI container smoke test:
    uv run python -m tests.fixtures.make_bundle /tmp/bundles
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from secops.data import schema as s
from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all
from secops.detection.bundle import BundleManifest
from secops.detection.features import FEATURE_SPEC_V1
from secops.detection.metrics import select_threshold

FIXTURE_DIR = Path(__file__).parent / "mini_cicids"


def make_fixture_bundles(out_dir: Path, n_estimators: int = 30) -> tuple[Path, Path]:
    df, _ = clean(read_all(FIXTURE_DIR, subdir=""), AttemptedPolicy.RELABEL_BENIGN)
    X = FEATURE_SPEC_V1.to_matrix(df)
    y = df["is_attack"].to_numpy()
    det = LGBMClassifier(n_estimators=n_estimators, random_state=0, verbose=-1).fit(X, y)
    thr = select_threshold(y, np.asarray(det.predict_proba(X))[:, 1], max_fpr=0.01)
    det_dir = out_dir / "detector"
    _write(det_dir, det, "secops-detector-fixture", X[:50])
    (det_dir / "threshold.json").write_text(
        json.dumps({"threshold": thr.threshold, "method": thr.method})
    )

    fam_rows = df[
        (df["is_attack"] == 1) & df["family"].astype(str).isin(s.FAMILY_CLASSIFIER_CLASSES)
    ]
    classes = list(s.FAMILY_CLASSIFIER_CLASSES)
    Xf = FEATURE_SPEC_V1.to_matrix(fam_rows)
    yf = np.searchsorted(classes, fam_rows["family"].astype(str).to_numpy())
    fam = LGBMClassifier(n_estimators=n_estimators, random_state=0, verbose=-1).fit(Xf, yf)
    fam_dir = out_dir / "family"
    _write(fam_dir, fam, "secops-family-fixture", Xf[:50])
    (fam_dir / "classes.json").write_text(json.dumps(classes))
    return det_dir, fam_dir


def _write(root: Path, estimator: LGBMClassifier, name: str, background: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(
        estimator,
        str(root / "model"),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    FEATURE_SPEC_V1.save(root / "feature_spec.json")
    pd.DataFrame(background, columns=FEATURE_SPEC_V1.names).to_parquet(
        root / "explainer_background.parquet", index=False
    )
    BundleManifest(
        model_name=name,
        version=0,
        run_id="fixture",
        alias="fixture",
        exported_at=datetime.now(UTC).isoformat(),
        feature_spec_version=FEATURE_SPEC_V1.version,
        model_kind="lightgbm",
        metrics={},
        tags={"source": "tests/fixtures/make_bundle.py"},
    ).to_json(root / "bundle.json")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    d, f = make_fixture_bundles(out)
    print(d, f)
