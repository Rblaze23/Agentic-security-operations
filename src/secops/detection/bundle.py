"""Model bundles: a self-contained directory that serving loads without any MLflow server.

Training keeps using the MLflow registry; `export_bundle` resolves an alias once and writes
everything inference needs. `load_bundle` reads it back and validates the pieces against each
other so a mismatched spec or class order fails at startup rather than at prediction time.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from secops.detection.features import FeatureSpec

MODEL_SUBDIR = "model"
MANIFEST_FILE = "bundle.json"
RUN_ARTIFACTS: tuple[tuple[str, bool], ...] = (
    ("feature_spec.json", True),
    ("threshold.json", False),
    ("classes.json", False),
    ("explainer_background.parquet", False),
)


class BundleError(ValueError):
    """The bundle directory is incomplete or internally inconsistent."""


@dataclass
class BundleManifest:
    model_name: str
    version: int
    run_id: str
    alias: str
    exported_at: str
    feature_spec_version: str
    model_kind: str
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def from_json(cls, path: Path) -> BundleManifest:
        return cls(**json.loads(path.read_text()))


@dataclass
class LoadedBundle:
    estimator: Any
    feature_spec: FeatureSpec
    threshold: float | None
    classes: list[str] | None
    background: np.ndarray | None
    manifest: BundleManifest
    path: Path


def export_bundle(model_name: str, out_dir: Path, alias: str = "champion") -> BundleManifest:
    """Materialise `models:/<model_name>@<alias>` plus its run artifacts into `out_dir`.

    The directory is replaced atomically-enough: everything is staged in a temporary directory
    next to the target, then swapped in, so a failed export never leaves a half-written bundle.
    """
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    run = mlflow.get_run(str(mv.run_id))
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        model_uri = f"models:/{model_name}@{alias}"
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, dst_path=str(staging / "_dl")
            )
        )
        shutil.move(str(downloaded), str(staging / MODEL_SUBDIR))
        shutil.rmtree(staging / "_dl", ignore_errors=True)
        present = {a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id)}
        for name, required in RUN_ARTIFACTS:
            if name in present:
                local = mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id, artifact_path=name, dst_path=str(staging / "_art")
                )
                shutil.move(str(local), str(staging / name))
            elif required:
                raise BundleError(f"run {run.info.run_id} has no {name} artifact")
        shutil.rmtree(staging / "_art", ignore_errors=True)
        spec = FeatureSpec.load(staging / "feature_spec.json")
        tags = {k: str(v) for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        manifest = BundleManifest(
            model_name=model_name,
            version=int(mv.version),
            run_id=str(mv.run_id),
            alias=alias,
            exported_at=datetime.now(UTC).isoformat(),
            feature_spec_version=spec.version,
            model_kind=tags.get("model", "unknown"),
            metrics={k: float(v) for k, v in run.data.metrics.items()},
            tags=tags,
        )
        manifest.to_json(staging / MANIFEST_FILE)
        _make_world_readable(staging)  # mkdtemp gives 0700; the container runs as a non-root user
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _swap_in(staging, out_dir)
    return manifest


def _make_world_readable(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o755)  # noqa: S103  # nosec B103 - read+exec for the container user
        for f in filenames:
            os.chmod(Path(dirpath) / f, 0o644)
    os.chmod(root, 0o755)  # noqa: S103  # nosec B103


def _swap_in(staging: Path, out_dir: Path) -> None:
    """Replace out_dir with staging; on failure the previous bundle is restored, never lost."""
    backup = out_dir.with_name(out_dir.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)
    had_previous = out_dir.exists()
    if had_previous:
        out_dir.rename(backup)
    try:
        staging.rename(out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if had_previous:
            backup.rename(out_dir)
        raise
    if had_previous:
        shutil.rmtree(backup)


def load_bundle(bundle_dir: Path) -> LoadedBundle:
    bundle_dir = Path(bundle_dir)
    spec_path = bundle_dir / "feature_spec.json"
    if not spec_path.exists():
        raise BundleError(f"{bundle_dir}: feature_spec.json is missing")
    manifest_path = bundle_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise BundleError(f"{bundle_dir}: {MANIFEST_FILE} is missing")
    estimator = mlflow.sklearn.load_model(str(bundle_dir / MODEL_SUBDIR))
    spec = FeatureSpec.load(spec_path)
    n_in = getattr(estimator, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(spec.names):
        raise BundleError(
            f"{bundle_dir}: estimator expects {int(n_in)} features but feature_spec.json "
            f"lists {len(spec.names)}"
        )
    threshold: float | None = None
    if (bundle_dir / "threshold.json").exists():
        threshold = float(json.loads((bundle_dir / "threshold.json").read_text())["threshold"])
    classes: list[str] | None = None
    if (bundle_dir / "classes.json").exists():
        classes = [str(c) for c in json.loads((bundle_dir / "classes.json").read_text())]
        est_classes = [int(c) for c in getattr(estimator, "classes_", [])]
        if est_classes != list(range(len(classes))):
            raise BundleError(
                f"{bundle_dir}: classes.json has {len(classes)} entries but the estimator's "
                f"classes_ are {est_classes}"
            )
    background: np.ndarray | None = None
    if (bundle_dir / "explainer_background.parquet").exists():
        bg = pd.read_parquet(bundle_dir / "explainer_background.parquet")
        background = spec.to_matrix(bg)
    return LoadedBundle(
        estimator=estimator,
        feature_spec=spec,
        threshold=threshold,
        classes=classes,
        background=background,
        manifest=BundleManifest.from_json(manifest_path),
        path=bundle_dir,
    )
