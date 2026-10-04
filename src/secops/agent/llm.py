"""Thin adapter over the official Anthropic SDK with record/replay.

Why not a chat-framework wrapper: the adapter exposes exactly the current API surface the agent
needs (adaptive thinking by default, `output_config.effort`, structured outputs through
`output_config.format`, prompt caching, usage accounting) and nothing else, and its replay mode
makes the whole graph testable without network or keys.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, Field

from secops.schemas.agent import UsageTotals

Mode = Literal["live", "record", "replay"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
DEFAULT_MAX_TOKENS = 16000


class LLMRefusalError(Exception):
    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f"model refused ({category}): {explanation}")
        self.category = category
        self.explanation = explanation


class LLMTruncatedError(Exception):
    """stop_reason == max_tokens: the structured answer is incomplete."""


class UnrecordedRequestError(Exception):
    """Replay mode has no fixture for this request (the prompt changed or the fixture is stale)."""


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMResponse(BaseModel):
    model: str
    stop_reason: str | None
    content: list[dict[str, Any]]
    usage: Usage
    parsed: dict[str, Any] | None = None
    request_id: str | None = None
    tool_uses: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def assistant_message(self) -> dict[str, Any]:
        """The turn to append to the history: content blocks echoed back unchanged."""
        return {"role": "assistant", "content": self.content}


def request_fingerprint(request: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()


# Keywords the structured-output / strict-tool grammar rejects (observed 2026-10-04:
# "For 'array' type, 'minItems' values other than 0 or 1 are not supported",
# "For 'string' type, format 'ipvanyaddress' is not supported", and every object must set
# additionalProperties: false). They are dropped from the schema sent to the API; Pydantic still
# enforces them client-side, so a violation surfaces as a ValidationError, never silently.
UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "pattern",
    }
)
SUPPORTED_FORMATS = frozenset({"date-time", "date", "time", "email", "uri", "uuid"})


def sanitize_schema(node: Any) -> Any:
    """A JSON schema the API's grammar accepts: constraints stripped, objects closed."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in UNSUPPORTED_SCHEMA_KEYWORDS:
                continue
            if k == "minItems" and isinstance(v, int) and v > 1:
                out[k] = 1
                continue
            if k == "format" and v not in SUPPORTED_FORMATS:
                continue
            out[k] = sanitize_schema(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [sanitize_schema(v) for v in node]
    return node


def _schema_for(model: type[BaseModel]) -> dict[str, Any]:
    schema: dict[str, Any] = sanitize_schema(model.model_json_schema())
    schema.setdefault("additionalProperties", False)
    return schema


class LLM:
    def __init__(
        self,
        model: str,
        effort: Effort = "medium",
        mode: Mode = "live",
        fixture_dir: Path | None = None,
        client: Any | None = None,
        usage: UsageTotals | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 120.0,
    ) -> None:
        if mode not in ("live", "record", "replay"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode in ("record", "replay") and fixture_dir is None:
            raise ValueError("fixture_dir is required for record and replay modes")
        self.model = model
        self.effort: Effort = effort
        self.mode: Mode = mode
        self.fixture_dir = Path(fixture_dir) if fixture_dir else None
        self.max_tokens = max_tokens
        self.usage = usage if usage is not None else UsageTotals()
        self._client = client
        self._timeout = timeout
        self._fixtures: dict[str, dict[str, Any]] | None = None

    # ---- request --------------------------------------------------------------------------
    def build_request(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        if tools:
            tools = [dict(t) for t in tools]
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            req["tools"] = tools
        if output_model is not None:
            req["output_config"]["format"] = {
                "type": "json_schema",
                "schema": _schema_for(output_model),
            }
        return req

    # ---- call -------------------------------------------------------------------------------
    def create(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        req = self.build_request(system, messages, tools, output_model)
        key = request_fingerprint(req)
        if self.mode == "replay":
            raw = self._load_fixture(key)
        else:
            raw = self._call(req)
            if self.mode == "record":
                self._store_fixture(key, req, raw)
        response = self._normalise(raw)
        self.usage.add(
            response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
            cache_write_tokens=response.usage.cache_write_tokens,
        )
        if response.stop_reason == "refusal":
            details = raw.get("stop_details") or {}
            raise LLMRefusalError(details.get("category"), details.get("explanation"))
        if response.stop_reason == "max_tokens":
            raise LLMTruncatedError("response hit max_tokens")
        if output_model is not None and response.stop_reason != "tool_use":
            response.parsed = output_model.model_validate_json(response.text).model_dump(
                mode="json"
            )
        return response

    def _client_or_default(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic(timeout=self._timeout, max_retries=2)
        return self._client

    def _call(self, req: dict[str, Any]) -> dict[str, Any]:
        message = self._client_or_default().messages.create(**req)
        raw: dict[str, Any] = message.model_dump(mode="json")
        raw["_request_id"] = getattr(message, "_request_id", None)
        return raw

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> LLMResponse:
        usage = raw.get("usage") or {}
        content = [dict(b) for b in raw.get("content", [])]
        return LLMResponse(
            model=str(raw.get("model")),
            stop_reason=raw.get("stop_reason"),
            content=content,
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
                cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            ),
            request_id=raw.get("_request_id"),
            tool_uses=[
                {"id": b["id"], "name": b["name"], "input": b.get("input") or {}}
                for b in content
                if b.get("type") == "tool_use"
            ],
        )

    # ---- fixtures ---------------------------------------------------------------------------
    def _store_fixture(self, key: str, req: dict[str, Any], raw: dict[str, Any]) -> None:
        assert self.fixture_dir is not None
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(self.fixture_dir.glob("*.json"))) + 1
        (self.fixture_dir / f"{n:03d}.json").write_text(
            json.dumps(
                {
                    "request_hash": key,
                    "model": self.model,
                    "captured_at": datetime.now(UTC).isoformat(),
                    "request": req,
                    "response": raw,
                },
                indent=1,
                default=str,
            )
        )
        self._fixtures = None

    def _load_fixture(self, key: str) -> dict[str, Any]:
        assert self.fixture_dir is not None
        if self._fixtures is None:
            self._fixtures = {}
            for p in sorted(self.fixture_dir.glob("*.json")):
                entry = json.loads(p.read_text())
                self._fixtures[entry["request_hash"]] = entry
        entry = self._fixtures.get(key)
        if entry is None:
            raise UnrecordedRequestError(
                f"no recorded response for request {key[:12]} in {self.fixture_dir} "
                "(the prompt changed or the fixture is stale; re-record the scenario)"
            )
        response: dict[str, Any] = entry["response"]
        return response
