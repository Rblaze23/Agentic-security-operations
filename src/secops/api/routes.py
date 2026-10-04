"""HTTP routes. Thin: validate, delegate to DetectorService, shape the response."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from secops.api.auth import require_api_key
from secops.api.detector import DetectorService
from secops.schemas.alert import Alert
from secops.schemas.api import BatchPredictResponse, HealthResponse, ModelInfo, PredictResponse
from secops.schemas.flow import BatchPredictRequest, FeatureMismatchError, PredictRequest

router = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])


def _service_or_none(request: Request) -> DetectorService | None:
    svc: DetectorService | None = getattr(request.app.state, "service", None)
    return svc


def get_service(request: Request) -> DetectorService:
    svc = _service_or_none(request)
    if svc is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="model bundles not loaded")
    return svc


Service = Annotated[DetectorService, Depends(get_service)]


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    svc = _service_or_none(request)
    if svc is None:
        return HealthResponse(status="starting", detector_loaded=False, family_loaded=False)
    versions = {"detector": svc.detector.manifest.version}
    if svc.family is not None:
        versions["family"] = svc.family.manifest.version
    return HealthResponse(
        status="ok" if svc.family is not None else "degraded",
        detector_loaded=True,
        family_loaded=svc.family is not None,
        bundle_versions=versions,
    )


@protected.get("/model", response_model=ModelInfo)
def model_info(svc: Service) -> ModelInfo:
    return svc.info()


def _predict(svc: DetectorService, items: list[PredictRequest]) -> BatchPredictResponse:
    try:
        predictions = svc.predict(items)
    except FeatureMismatchError as e:  # only client-caused name drift maps to 422
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e
    alerts = [
        Alert.from_prediction(req, pred)
        for req, pred in zip(items, predictions, strict=True)
        if pred.is_alert
    ]
    return BatchPredictResponse(predictions=predictions, alerts=alerts)


@protected.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, svc: Service) -> PredictResponse:
    out = _predict(svc, [body])
    return PredictResponse(
        prediction=out.predictions[0], alert=out.alerts[0] if out.alerts else None
    )


@protected.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest, svc: Service) -> BatchPredictResponse:
    return _predict(svc, body.items)
