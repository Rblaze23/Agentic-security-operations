"""Structured request logging with a per-request id."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from secops.api.auth import UNAUTHORIZED_HEADERS, key_is_valid

PROTECTED_PREFIXES = ("/predict", "/model", "/docs", "/openapi.json")

log = logging.getLogger("secops.api")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = rid
    start = time.perf_counter()
    response = await _guarded(request, call_next)
    response.headers["X-Request-ID"] = rid
    log.info(
        "request",
        extra={
            "extra": {
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        },
    )
    return response


async def _guarded(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Refuse before the body is read: oversized payloads (413) and unauthenticated calls to
    protected routes (401). The route dependency re-checks the key (defence in depth)."""
    settings = request.app.state.settings
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > settings.max_body_bytes:
        return JSONResponse(status_code=413, content={"detail": "request body too large"})
    if request.url.path.startswith(PROTECTED_PREFIXES) and not key_is_valid(
        request.headers.get("X-API-Key"), settings.keys
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "missing or invalid API key"},
            headers=UNAUTHORIZED_HEADERS,
        )
    try:
        return await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"extra": {"request_id": request.state.request_id}})
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "request_id": request.state.request_id},
        )
