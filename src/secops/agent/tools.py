"""Expose the Phase 3 registry to the model and execute its calls as Evidence.

Every tool call, successful or not, becomes an `Evidence` record with a stable id. The model
receives a compact envelope (id, tool, summary, truncated data); the full payload stays in the
graph state for the critic and the report.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from secops.agent.llm import sanitize_schema
from secops.schemas.agent import MAX_SUMMARY_CHARS, Evidence, ToolCallRecord
from secops.tools.registry import ToolRegistry

log = logging.getLogger(__name__)
DEFAULT_BUDGET = 12
DEFAULT_MAX_PAYLOAD_CHARS = 4000


# ---- definitions ---------------------------------------------------------------------------
def _is_strict_compatible(schema: dict[str, Any]) -> bool:
    props = set(schema.get("properties", {}))
    required = set(schema.get("required", []))
    return bool(props) and props == required


def tool_definitions(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Anthropic tool definitions from the registry's Pydantic input models.

    `strict` is set only when every property is required (the API's strict mode needs that);
    inputs are re-validated by Pydantic on execution regardless.
    """
    defs: list[dict[str, Any]] = []
    for spec in registry.all():
        schema = sanitize_schema(spec.input_schema())
        schema["additionalProperties"] = False
        definition: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "input_schema": schema,
        }
        if _is_strict_compatible(schema):
            definition["strict"] = True
        defs.append(definition)
    return defs


# ---- summaries -----------------------------------------------------------------------------
def _clip(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarize(tool: str, output: BaseModel) -> str:
    """Deterministic one-paragraph summary of a tool output for the model and the report."""
    d = output.model_dump(mode="json")
    if tool == "search_events":
        n = len(d.get("events", []))
        return _clip(
            f"{d.get('total_matched', 0)} flows matched; returning {n}"
            + (" (truncated)" if d.get("truncated") else "")
            + (
                f"; first {d['events'][0]['timestamp']} last {d['events'][-1]['timestamp']}"
                if n
                else ""
            )
            + "."
        )
    if tool == "get_related_events":
        if d.get("status") != "found":
            return "event not found."
        s, dst, pair = d["same_source"], d["same_destination"], d["same_pair"]
        ports = ", ".join(
            f"{p['port']}x{p['count']}" for p in s.get("top_destination_ports", [])[:5]
        )
        return _clip(
            f"±{d['window_minutes']} min: same source {s['count']} flows to "
            f"{s['distinct_destination_ips']} IPs / {s['distinct_destination_ports']} ports "
            f"(top ports {ports or 'none'}); same destination {dst['count']} flows; "
            f"same pair {pair['count']} flows, fwd bytes {pair['total_fwd_bytes']}, "
            f"bwd bytes {pair['total_bwd_bytes']}."
        )
    if tool == "get_asset":
        a = d.get("asset")
        if not a:
            return "no asset record for this IP."
        return _clip(
            f"{a['hostname']} ({a['ip']}): {a['role']}, {a['os']}, criticality {a['criticality']}, "
            f"zone {a['zone']}, services {', '.join(a['services'][:8]) or 'none'}"
            + (f"; notes: {a['notes']}" if a.get("notes") else "")
            + "."
        )
    if tool == "enrich_ip":
        return _clip(
            f"{d['ip']}: {'private' if d['is_private'] else 'public'}, "
            f"zone {d['zone'] or 'unknown'}, known attacker: {d['known_attacker']}"
            + (f" ({d['attacker_notes']})" if d.get("attacker_notes") else "")
            + (
                f"; asset {d['asset']['hostname']} ({d['asset']['criticality']})"
                if d.get("asset")
                else ""
            )
            + "."
        )
    if tool == "lookup_attack_technique":
        if d.get("status") != "found":
            return "no matching ATT&CK technique."
        parts = [
            f"{t['technique_id']} {t['name']} [{', '.join(t['tactics'])}]" for t in d["techniques"]
        ]
        return _clip(f"ATT&CK v{d['attack_version']}: " + "; ".join(parts) + ".")
    if tool == "lookup_cve":
        if d.get("status") != "found":
            return f"NVD: {d.get('status')}."
        parts = [
            f"{r['cve_id']} CVSS {r['cvss_v3_score']} {r['cvss_v3_severity'] or ''}: "
            f"{r['description'][:160]}"
            for r in d["records"]
        ]
        return _clip(("cached. " if d.get("cached") else "") + " | ".join(parts))
    if tool == "predict_attack":
        preds = d.get("predictions", [])
        alerts = sum(1 for p in preds if p.get("is_alert"))
        return _clip(
            f"{len(preds)} flows scored, {alerts} above threshold"
            + (f"; missing ids {d['missing_event_ids']}" if d.get("missing_event_ids") else "")
            + "."
        )
    return _clip(json.dumps(d, sort_keys=True, default=str))


# ---- execution -----------------------------------------------------------------------------
def evidence_data(ev: Evidence, max_chars: int = DEFAULT_MAX_PAYLOAD_CHARS) -> str:
    """The data block exactly as the investigator saw it (the critic must see the same)."""
    data = json.dumps(ev.payload, sort_keys=True, default=str)
    if len(data) > max_chars:
        data = data[:max_chars] + f"… [truncated, {len(data)} chars total]"
    return data


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        budget: int = DEFAULT_BUDGET,
        max_payload_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
        tracer: Any | None = None,
    ) -> None:
        self.registry = registry
        self.remaining = budget
        self.max_payload_chars = max_payload_chars
        self.tracer = tracer  # secops.observability.Tracer
        self.records: list[ToolCallRecord] = []
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"E{self._counter}"

    def execute(self, tool_use: dict[str, Any]) -> tuple[dict[str, Any], Evidence]:
        name = str(tool_use.get("name"))
        args = dict(tool_use.get("input") or {})
        evidence_id = self._next_id()
        start = time.perf_counter()
        if self.remaining <= 0:
            ev = self._error(
                evidence_id,
                name,
                args,
                "tool budget exhausted; finish with the evidence collected so far",
                start,
            )
            return self._result_block(tool_use, ev, error=True), self._record(
                ev, "budget_exhausted"
            )
        self.remaining -= 1
        try:
            spec = self.registry.get(name)
        except KeyError:
            ev = self._error(
                evidence_id,
                name,
                args,
                f"unknown tool {name!r}; available: {', '.join(self.registry.names())}",
                start,
            )
            return self._result_block(tool_use, ev, error=True), self._record(ev, "error")
        try:
            output = spec.invoke(args)
        except ValidationError as e:
            fields = ", ".join(
                ".".join(str(x) for x in err["loc"]) or "(root)" for err in e.errors()
            )
            problems = "; ".join(
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            ev = self._error(
                evidence_id, name, args, f"invalid arguments ({fields}): {problems}", start
            )
            return self._result_block(tool_use, ev, error=True), self._record(ev, "error")
        except Exception:
            log.exception("tool %s failed", name)
            ev = self._error(
                evidence_id,
                name,
                args,
                f"tool {name} failed internally; the result is unavailable",
                start,
            )
            return self._result_block(tool_use, ev, error=True), self._record(ev, "error")
        payload = output.model_dump(mode="json")
        ev = Evidence(
            evidence_id=evidence_id,
            tool=name,
            arguments=args,
            kind="tool_result",
            summary=summarize(name, output),
            payload=payload,
            retrieved_at=datetime.now(UTC),
            untrusted_text=spec.external_source is not None,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return self._result_block(tool_use, ev, error=False), self._record(ev, "ok")

    # ---- helpers ------------------------------------------------------------------------
    @staticmethod
    def _error(
        evidence_id: str, name: str, args: dict[str, Any], message: str, start: float
    ) -> Evidence:
        return Evidence(
            evidence_id=evidence_id,
            tool=name,
            arguments=args,
            kind="tool_error",
            summary=_clip(message),
            payload={"error": _clip(message)},
            retrieved_at=datetime.now(UTC),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def _record(self, ev: Evidence, status: str) -> Evidence:
        self.records.append(
            ToolCallRecord(
                tool=ev.tool,
                arguments=ev.arguments,
                evidence_id=ev.evidence_id,
                status=status,  # type: ignore[arg-type]
                latency_ms=ev.latency_ms,
            )
        )
        if self.tracer is not None:
            self.tracer.tool_call(ev.tool, ev.evidence_id, status, ev.latency_ms)
        return ev

    def _result_block(self, tool_use: dict[str, Any], ev: Evidence, error: bool) -> dict[str, Any]:
        data = evidence_data(ev, self.max_payload_chars)
        envelope = {
            "evidence_id": ev.evidence_id,
            "tool": ev.tool,
            "kind": ev.kind,
            "summary": ev.summary,
            "untrusted_text": ev.untrusted_text,
            "data": data,
        }
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use.get("id"),
            "content": json.dumps(envelope, default=str),
        }
        if error:
            block["is_error"] = True
        return block
