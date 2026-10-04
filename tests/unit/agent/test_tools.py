from typing import Any

import pytest

from secops.agent.tools import ToolExecutor, summarize, tool_definitions
from secops.schemas.agent import Evidence
from secops.schemas.tools import Asset, AssetResult
from secops.tools.registry import ToolRegistry


def test_tool_definitions_have_schemas_and_strictness_rule(registry: ToolRegistry) -> None:
    defs = tool_definitions(registry)
    assert {d["name"] for d in defs} == set(registry.names())
    for d in defs:
        assert d["description"] and d["input_schema"]["type"] == "object"
        assert d["input_schema"].get("additionalProperties") is False
        props = set(d["input_schema"].get("properties", {}))
        required = set(d["input_schema"].get("required", []))
        assert d.get("strict", False) == (props == required and bool(props)), d["name"]
    assert next(d for d in defs if d["name"] == "get_asset")["strict"] is True
    assert next(d for d in defs if d["name"] == "search_events").get("strict", False) is False


def test_execute_returns_evidence_and_tool_result(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, budget=12)
    block, ev = ex.execute({"id": "tu_1", "name": "get_asset", "input": {"ip": "192.168.10.50"}})
    assert isinstance(ev, Evidence) and ev.evidence_id == "E1" and ev.kind == "tool_result"
    assert ev.tool == "get_asset" and ev.payload["status"] == "found"
    assert (
        block["type"] == "tool_result"
        and block["tool_use_id"] == "tu_1"
        and not block.get("is_error")
    )
    assert '"evidence_id": "E1"' in block["content"]
    assert ex.remaining == 11 and ex.records[0].status == "ok" and ex.records[0].latency_ms >= 0
    assert "web-server-16" in ev.summary and len(ev.summary) <= 600


def test_validation_error_becomes_tool_error_naming_the_field(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, budget=12)
    block, ev = ex.execute(
        {"id": "tu_2", "name": "get_asset", "input": {"ip": "not-an-ip", "bogus": 1}}
    )
    assert ev.kind == "tool_error" and block["is_error"] is True
    assert "ip" in ev.summary and "bogus" in ev.summary
    assert ex.remaining == 11  # a rejected call still consumes budget


def test_unknown_tool_and_exception_become_tool_error(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    ex = ToolExecutor(registry, budget=12)
    block, ev = ex.execute({"id": "tu_3", "name": "launch_missiles", "input": {}})
    assert ev.kind == "tool_error" and "unknown tool" in ev.summary and block["is_error"] is True

    spec = registry.get("enrich_ip")

    def boom(_inp: Any) -> Any:
        raise RuntimeError("secret stack trace /home/user/x.py")

    monkeypatch.setattr(spec, "run", boom)
    block, ev = ex.execute({"id": "tu_4", "name": "enrich_ip", "input": {"ip": "8.8.8.8"}})
    assert (
        ev.kind == "tool_error" and "secret" not in ev.summary and "/home" not in block["content"]
    )
    assert ex.records[-1].status == "error"


def test_tool_budget_enforced(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, budget=2)
    for i in range(2):
        ex.execute({"id": f"tu_{i}", "name": "get_asset", "input": {"ip": "192.168.10.3"}})
    block, ev = ex.execute({"id": "tu_x", "name": "get_asset", "input": {"ip": "192.168.10.3"}})
    assert ex.remaining == 0 and block["is_error"] is True
    assert "budget" in block["content"] and ev.kind == "tool_error"
    assert ex.records[-1].status == "budget_exhausted" and ev.evidence_id == "E3"


def test_external_text_marks_evidence_untrusted(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, budget=12)
    _, cve = ex.execute({"id": "a", "name": "lookup_cve", "input": {"cve_id": "CVE-2014-0160"}})
    _, asset = ex.execute({"id": "b", "name": "get_asset", "input": {"ip": "192.168.10.50"}})
    assert cve.untrusted_text is True and asset.untrusted_text is False
    assert "CVE-2014-0160" in cve.summary and "7.5" in cve.summary


def test_summaries_are_deterministic_and_bounded() -> None:
    asset = AssetResult(
        status="found",
        asset=Asset(
            ip="1.1.1.1",
            hostname="h",
            role="r",
            os="o",
            services=["x"] * 200,
            criticality="high",
            zone="z",
        ),
        source="s",
    )
    s1, s2 = summarize("get_asset", asset), summarize("get_asset", asset)
    assert s1 == s2 and len(s1) <= 600


def test_payload_in_tool_result_is_truncated(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, budget=12, max_payload_chars=200)
    block, ev = ex.execute(
        {"id": "c", "name": "lookup_cve", "input": {"keyword": "slowloris", "max_results": 5}}
    )
    assert len(block["content"]) < 1200  # envelope + truncated data
    assert "truncated" in block["content"]
    assert len(ev.payload["records"]) >= 1  # the full payload stays in state
