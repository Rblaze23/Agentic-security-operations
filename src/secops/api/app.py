"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse

from secops import __version__
from secops.api.auth import require_api_key
from secops.api.detector import DetectorService
from secops.api.logging import configure_logging, request_id_middleware
from secops.api.routes import protected, router
from secops.api.settings import ApiSettings, get_api_settings

log = logging.getLogger("secops.api")


def create_app(
    service: DetectorService | None = None, settings: ApiSettings | None = None
) -> FastAPI:
    settings = settings or get_api_settings()
    configure_logging(settings.log_level)
    if not settings.keys:
        log.error("SECOPS_API_KEYS is empty: every protected route will answer 401")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.service is None:
            family = settings.family_dir if settings.family_dir.exists() else None
            if family is None:
                log.warning(
                    "family bundle not found; serving degraded (no attack-family labels)",
                    extra={"extra": {"expected": str(settings.family_dir)}},
                )
            app.state.service = DetectorService.from_dirs(
                settings.detector_dir, family, k=settings.top_k
            )
            log.info("bundles loaded", extra={"extra": {"model_dir": str(settings.model_dir)}})
        yield

    app = FastAPI(
        title="SecOps detection API",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.service = service
    app.middleware("http")(request_id_middleware)
    app.include_router(router)
    app.include_router(protected)

    @app.get("/openapi.json", include_in_schema=False, dependencies=[Depends(require_api_key)])
    def openapi() -> JSONResponse:
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False, dependencies=[Depends(require_api_key)])
    def docs() -> HTMLResponse:
        return get_swagger_ui_html(openapi_url="/openapi.json", title=app.title)

    @app.exception_handler(RequestValidationError)
    async def validation_failed(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not echo `input`: it may be a non-finite float (NaN/Infinity parse as Python floats
        # but are not JSON) or a large payload. Locations and messages are enough to act on.
        errors = [
            {"type": e.get("type"), "loc": list(e.get("loc", ())), "msg": e.get("msg")}
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        log.exception("unhandled error", extra={"extra": {"request_id": rid}})
        return JSONResponse(
            status_code=500, content={"detail": "internal error", "request_id": rid}
        )

    return app
