from datetime import UTC, datetime

from secops.schemas.api import BundleInfo, HealthResponse, ModelInfo


def test_health_and_model_info_shapes() -> None:
    h = HealthResponse(status="starting", detector_loaded=False, family_loaded=False)
    assert h.bundle_versions == {}
    b = BundleInfo(
        model_name="secops-detector",
        version=1,
        run_id="abc",
        alias="champion",
        exported_at=datetime(2026, 10, 3, tzinfo=UTC),
        feature_spec_version="v1-noport",
        model_kind="lightgbm",
        metrics={"test_pr_auc": 0.9999},
        tags={"git_sha": "deadbeef"},
    )
    m = ModelInfo(
        detector=b, family=None, feature_spec_version="v1-noport", threshold=0.0002, top_k=5
    )
    assert m.model_dump()["detector"]["metrics"]["test_pr_auc"] == 0.9999
