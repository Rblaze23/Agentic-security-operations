"""Per-API-key token bucket. In-process (one bucket map per worker); Phase 6 documents that a
shared store is the next step when more than one replica serves a key."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from secops.api.auth import settings_from_app
from secops.api.settings import ApiSettings


class TokenBucket:
    def __init__(self, rate_per_minute: int, now: Callable[[], float] = time.monotonic) -> None:
        self.capacity = float(rate_per_minute)
        self.refill_per_second = rate_per_minute / 60.0
        self._now = now
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill_at)

    def take(self, key: str) -> tuple[bool, float]:
        """Consume one token for `key`. Returns (allowed, seconds until the next token)."""
        with self._lock:
            now = self._now()
            tokens, last = self._state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            return False, (1.0 - tokens) / self.refill_per_second


def bucket_from_app(request: Request) -> TokenBucket:
    bucket = getattr(request.app.state, "rate_limiter", None)
    if bucket is None:
        settings: ApiSettings = request.app.state.settings
        bucket = TokenBucket(settings.rate_limit_per_minute)
        request.app.state.rate_limiter = bucket
    return bucket


def rate_limited(
    settings: Annotated[ApiSettings, Depends(settings_from_app)],
    bucket: Annotated[TokenBucket, Depends(bucket_from_app)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """429 with Retry-After when the presented key has no token left. Runs after the key check,
    so the bucket is keyed by a valid key; an absent key is keyed as "" and still limited."""
    if settings.rate_limit_per_minute <= 0:
        return
    allowed, wait = bucket.take(x_api_key or "")
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded for this API key",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )
