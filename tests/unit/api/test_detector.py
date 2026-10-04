from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secops.api.detector import DetectorService
from secops.data.schema import FAMILY_CLASSIFIER_CLASSES
from secops.detection.bundle import BundleError
from secops.detection.features import FeatureSpec
from tests.unit.api.conftest import to_request


def test_alert_flag_follows_bundle_threshold(
    service: DetectorService, fixture_flows: pd.DataFrame
) -> None:
    reqs = [to_request(r) for _, r in fixture_flows.head(200).iterrows()]
    preds = service.predict(reqs)
    assert len(preds) == 200
    for p in preds:
        assert p.is_alert == (p.attack_probability >= p.threshold)
        assert p.threshold == service.detector.threshold
        assert p.model_name == "secops-detector-fixture" and p.feature_spec_version == "v1-noport"
    assert any(p.is_alert for p in preds) and any(not p.is_alert for p in preds)


def test_family_only_on_alerts(service: DetectorService, fixture_flows: pd.DataFrame) -> None:
    preds = service.predict([to_request(r) for _, r in fixture_flows.head(300).iterrows()])
    for p in preds:
        if p.is_alert:
            assert p.predicted_family in FAMILY_CLASSIFIER_CLASSES
            assert p.family_probabilities is not None
            assert set(p.family_probabilities) == set(FAMILY_CLASSIFIER_CLASSES)
            assert sum(p.family_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
            assert p.predicted_family == max(p.family_probabilities, key=p.family_probabilities.get)
        else:
            assert p.predicted_family is None and p.family_probabilities is None


def test_top_contributions_sorted_and_null_for_nan(
    service: DetectorService, fixture_flows: pd.DataFrame
) -> None:
    req = to_request(fixture_flows.iloc[0])
    req.features["Flow Bytes/s"] = None
    (p,) = service.predict([req])
    assert len(p.top_contributions) == 5
    mags = [abs(c.shap_value) for c in p.top_contributions]
    assert mags == sorted(mags, reverse=True)
    for c in p.top_contributions:
        if c.feature == "Flow Bytes/s":
            assert c.value is None
    assert p.latency_ms >= 0


def test_batch_equals_single_calls(service: DetectorService, fixture_flows: pd.DataFrame) -> None:
    reqs = [
        to_request(r, event_id=f"e{i}") for i, (_, r) in enumerate(fixture_flows.head(5).iterrows())
    ]
    batch = service.predict(reqs)
    singles = [service.predict([r])[0] for r in reqs]
    for b, s_ in zip(batch, singles, strict=True):
        assert b.attack_probability == pytest.approx(s_.attack_probability)
        assert b.event_id == s_.event_id and b.prediction_id != s_.prediction_id


def test_feature_name_drift_raises_value_error(
    service: DetectorService, fixture_flows: pd.DataFrame
) -> None:
    req = to_request(fixture_flows.iloc[0])
    req.features.pop("Flow Duration")
    with pytest.raises(ValueError, match="missing"):
        service.predict([req])


def test_service_rejects_family_with_other_feature_spec(
    bundle_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    import shutil

    det, fam = bundle_dirs
    bad = tmp_path / "fam_bad"
    shutil.copytree(fam, bad)
    spec = FeatureSpec.load(bad / "feature_spec.json")
    FeatureSpec(names=spec.names, version="v9-other").save(bad / "feature_spec.json")
    with pytest.raises(BundleError, match="feature spec"):
        DetectorService.from_dirs(det, bad)


def test_service_requires_threshold_and_classes(
    bundle_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    import shutil

    det, fam = bundle_dirs
    no_thr = tmp_path / "det_no_thr"
    shutil.copytree(det, no_thr)
    (no_thr / "threshold.json").unlink()
    with pytest.raises(BundleError, match="threshold"):
        DetectorService.from_dirs(no_thr, fam)
    no_cls = tmp_path / "fam_no_cls"
    shutil.copytree(fam, no_cls)
    (no_cls / "classes.json").unlink()
    with pytest.raises(BundleError, match="classes"):
        DetectorService.from_dirs(det, no_cls)


def test_info_reports_bundle_manifests(service: DetectorService) -> None:
    info = service.info()
    assert info.detector.model_name == "secops-detector-fixture" and info.family is not None
    assert info.feature_spec_version == "v1-noport" and info.top_k == 5
    assert info.threshold == service.detector.threshold


def test_batch_contributions_match_single(
    service: DetectorService, fixture_flows: pd.DataFrame
) -> None:
    from secops.detection.explain import top_k_contributions, top_k_contributions_batch

    X = service.detector.feature_spec.to_matrix(fixture_flows.head(4))
    names = service.detector.feature_spec.names
    batch = top_k_contributions_batch(service.explainer, X, names, k=3)
    for i in range(4):
        single = top_k_contributions(service.explainer, X[i], names, k=3)
        assert [c.feature for c in batch[i]] == [c.feature for c in single]
        assert np.allclose([c.shap_value for c in batch[i]], [c.shap_value for c in single])
