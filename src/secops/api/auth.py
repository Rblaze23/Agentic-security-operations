"""API-key authentication as a FastAPI dependency (constant-time comparison, fail closed)."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from secops.api.settings import ApiSettings


def settings_from_app(request: Request) -> ApiSettings:
    settings: ApiSettings = request.app.state.settings
    return settings


UNAUTHORIZED_HEADERS = {"WWW-Authenticate": 'ApiKey realm="secops"'}


def key_is_valid(presented: str | None, keys: list[str]) -> bool:
    """Constant-time comparison on bytes, so non-ASCII input is simply wrong, not a 500."""
    candidate = (presented or "").encode("utf-8", "surrogateescape")
    return any(hmac.compare_digest(candidate, k.encode("utf-8")) for k in keys)


def require_api_key(
    settings: Annotated[ApiSettings, Depends(settings_from_app)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if not key_is_valid(x_api_key, settings.keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers=UNAUTHORIZED_HEADERS,
        )
