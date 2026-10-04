"""Tracing for investigations: one trace per investigation with a span per model call and per
tool call. Langfuse when its keys are configured and the package imports; a silent NullTracer
otherwise. A tracer can never break an investigation: every call is guarded."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


class Tracer(Protocol):
    def start(self, investigation_id: str, alert_id: str, prompt_version: str) -> None: ...
    def llm_call(
        self,
        model: str,
        request_id: str | None,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        latency_ms: float,
        cost_usd: float,
    ) -> None: ...
    def tool_call(self, name: str, evidence_id: str, status: str, latency_ms: float) -> None: ...
    def end(self, verdict: str, severity: str, cost_usd: float) -> None: ...


class NullTracer:
    def start(self, investigation_id: str, alert_id: str, prompt_version: str) -> None:
        return None

    def llm_call(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tool_call(self, name: str, evidence_id: str, status: str, latency_ms: float) -> None:
        return None

    def end(self, verdict: str, severity: str, cost_usd: float) -> None:
        return None


@dataclass
class RecordingTracer:
    """Keeps every event in memory (tests, and the CLI's --trace-dump)."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def start(self, investigation_id: str, alert_id: str, prompt_version: str) -> None:
        self.events.append(
            {
                "type": "start",
                "investigation_id": investigation_id,
                "alert_id": alert_id,
                "prompt_version": prompt_version,
            }
        )

    def llm_call(
        self,
        model: str,
        request_id: str | None,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        self.events.append(
            {
                "type": "llm",
                "model": model,
                "request_id": request_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            }
        )

    def tool_call(self, name: str, evidence_id: str, status: str, latency_ms: float) -> None:
        self.events.append(
            {
                "type": "tool",
                "name": name,
                "evidence_id": evidence_id,
                "status": status,
                "latency_ms": latency_ms,
            }
        )

    def end(self, verdict: str, severity: str, cost_usd: float) -> None:
        self.events.append(
            {"type": "end", "verdict": verdict, "severity": severity, "cost_usd": cost_usd}
        )


class SafeTracer:
    """Wraps any tracer so that an exception inside it is logged once and otherwise ignored."""

    def __init__(self, inner: Tracer) -> None:
        self._inner = inner
        self._failed = False

    def _guard(self, method: str, *args: Any, **kwargs: Any) -> None:
        if self._failed:
            return
        try:
            getattr(self._inner, method)(*args, **kwargs)
        except Exception as e:  # tracing must never break an investigation
            self._failed = True
            log.warning("tracer disabled after error in %s: %s: %s", method, type(e).__name__, e)

    def start(self, investigation_id: str, alert_id: str, prompt_version: str) -> None:
        self._guard("start", investigation_id, alert_id, prompt_version)

    def llm_call(self, *args: Any, **kwargs: Any) -> None:
        self._guard("llm_call", *args, **kwargs)

    def tool_call(self, name: str, evidence_id: str, status: str, latency_ms: float) -> None:
        self._guard("tool_call", name, evidence_id, status, latency_ms)

    def end(self, verdict: str, severity: str, cost_usd: float) -> None:
        self._guard("end", verdict, severity, cost_usd)


class LangfuseTracer:
    """Langfuse SDK v4 (OpenTelemetry-based) wiring: one root observation per investigation,
    a child generation per model call, a child span per tool call. The client is typed as Any
    because the SDK is an optional dependency."""

    def __init__(self, public_key: str, secret_key: str, host: str | None = None) -> None:
        from langfuse import Langfuse  # optional dependency

        kwargs: dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
        if host:
            kwargs["host"] = host
        self._client: Any = Langfuse(**kwargs)
        self._root: Any = None

    def start(self, investigation_id: str, alert_id: str, prompt_version: str) -> None:
        self._root = self._client.start_observation(
            name="investigation",
            as_type="span",
            input={"alert_id": alert_id},
            metadata={"investigation_id": investigation_id, "prompt_version": prompt_version},
        )

    def llm_call(
        self,
        model: str,
        request_id: str | None,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        if self._root is None:
            return
        gen = self._root.start_observation(
            name="messages.create",
            as_type="generation",
            model=model,
            metadata={"request_id": request_id, "latency_ms": latency_ms},
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "cache_read_input_tokens": cache_read_tokens,
            },
            cost_details={"total": cost_usd},
        )
        gen.end()

    def tool_call(self, name: str, evidence_id: str, status: str, latency_ms: float) -> None:
        if self._root is None:
            return
        span = self._root.start_observation(
            name=name,
            as_type="span",
            metadata={"evidence_id": evidence_id, "status": status, "latency_ms": latency_ms},
        )
        span.end()

    def end(self, verdict: str, severity: str, cost_usd: float) -> None:
        if self._root is None:
            return
        self._root.update(output={"verdict": verdict, "severity": severity, "cost_usd": cost_usd})
        self._root.end()
        self._root = None
        self._client.flush()


def get_tracer(env: dict[str, str] | None = None) -> Tracer:
    """Langfuse when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and the SDK imports;
    NullTracer otherwise. Always wrapped in SafeTracer."""
    env = env if env is not None else dict(os.environ)
    public, secret = env.get("LANGFUSE_PUBLIC_KEY"), env.get("LANGFUSE_SECRET_KEY")
    if public and secret:
        try:
            return SafeTracer(LangfuseTracer(public, secret, env.get("LANGFUSE_HOST") or None))
        except Exception as e:  # package missing or client refused
            log.warning(
                "Langfuse tracing unavailable (%s: %s); tracing disabled", type(e).__name__, e
            )
    return SafeTracer(NullTracer())
