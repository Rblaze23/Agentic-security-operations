from __future__ import annotations

from typing import Any

from secops.agent.graph import AgentDeps, run_investigation
from secops.agent.llm import LLM
from secops.observability import NullTracer, RecordingTracer, SafeTracer, get_tracer
from secops.tools.registry import ToolRegistry
from tests.unit.agent.test_graph import _alert, _critic_ok, _draft_msg, _plan_msg, _tool_call
from tests.unit.agent.test_llm import FakeClient


def _deps(registry: ToolRegistry, flows: Any, tracer: Any) -> tuple[Any, AgentDeps]:
    alert = _alert(flows)
    msgs = [
        _plan_msg(),
        _tool_call(
            "t1", "get_related_events", {"event_id": int(alert.event_id), "window_minutes": 5}
        ),
        _tool_call("t2", "get_asset", {"ip": str(alert.metadata.destination_ip)}),
        _draft_msg(findings=[{"kind": "observed", "statement": "burst", "evidence_ids": ["E1"]}]),
    ]
    deps = AgentDeps(
        registry=registry,
        investigator=LLM("claude-opus-5-5", client=FakeClient(msgs)),
        critic=LLM("claude-sonnet-5-5", client=FakeClient([_critic_ok()])),
        tool_budget=12,
        tracer=tracer,
    )
    return alert, deps


def test_recording_tracer_sees_the_whole_investigation(registry: ToolRegistry, flows: Any) -> None:
    tracer = RecordingTracer()
    alert, deps = _deps(registry, flows, tracer)
    result = run_investigation(alert, deps)
    types = [e["type"] for e in tracer.events]
    assert types[0] == "start" and types[-1] == "end"
    assert types.count("llm") == result.usage.calls == 5
    assert [e["name"] for e in tracer.events if e["type"] == "tool"] == [
        "get_related_events",
        "get_asset",
    ]
    assert sum(e["cost_usd"] for e in tracer.events if e["type"] == "llm") == result.usage.cost_usd
    assert tracer.events[0]["investigation_id"] == result.investigation_id
    assert tracer.events[-1]["verdict"] == result.report.verdict


class _Exploding:
    def start(self, *a: Any, **k: Any) -> None:
        raise RuntimeError("tracing backend down")

    llm_call = tool_call = end = start


def test_tracing_failure_is_swallowed(registry: ToolRegistry, flows: Any) -> None:
    alert, deps = _deps(registry, flows, SafeTracer(_Exploding()))
    traced = run_investigation(alert, deps)
    alert2, deps2 = _deps(registry, flows, NullTracer())
    plain = run_investigation(alert2, deps2)
    assert traced.report.model_dump(exclude={"alert_id"}) == plain.report.model_dump(
        exclude={"alert_id"}
    )


def test_get_tracer_without_keys_is_null() -> None:
    tracer = get_tracer({})
    assert isinstance(tracer, SafeTracer) and isinstance(tracer._inner, NullTracer)
    tracer = get_tracer({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})
    assert isinstance(tracer, SafeTracer)  # Langfuse if importable, else Null; never raises
