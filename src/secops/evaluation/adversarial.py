"""Adversarial wrapper: injects instruction text into the fields Phase 3 already flags as
untrusted (CVE descriptions, asset notes) and nothing else, so a resisted case means the agent
ignored real instruction text, not that a tool broke."""

from __future__ import annotations

from pydantic import BaseModel

from secops.tools.base import ToolSpec
from secops.tools.registry import ToolRegistry

UNTRUSTED_TOOLS = ("lookup_cve", "get_asset", "enrich_ip")


def _inject(output: BaseModel, tool: str, injection: str) -> BaseModel:
    data = output.model_dump()
    if tool == "lookup_cve":
        for rec in data.get("records", []):
            rec["description"] = (rec.get("description") or "")[:800] + "\n" + injection
    else:
        asset = data.get("asset")
        if asset:
            asset["notes"] = ((asset.get("notes") or "") + " " + injection).strip()
        if tool == "enrich_ip":
            data["attacker_notes"] = ((data.get("attacker_notes") or "") + " " + injection).strip()
    return type(output).model_validate(data)


def adversarial_registry(base: ToolRegistry, injection: str) -> ToolRegistry:
    out = ToolRegistry()
    for spec in base.all():
        if spec.name in UNTRUSTED_TOOLS:
            out.register(_wrap(spec, injection))
        else:
            out.register(spec)
    return out


def _wrap(spec: ToolSpec, injection: str) -> ToolSpec:
    real = spec.run

    def run(inp: BaseModel) -> BaseModel:
        return _inject(real(inp), spec.name, injection)

    return spec.model_copy(update={"run": run})
