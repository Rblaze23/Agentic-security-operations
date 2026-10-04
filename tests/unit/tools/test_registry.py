from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from secops.api.detector import DetectorService
from secops.tools.detector import DetectorTool
from secops.tools.registry import EXPECTED_TOOLS, ToolRegistry, build_registry
from tests.unit.tools.test_cve import FixtureFetcher

SUBSET = Path(__file__).resolve().parents[2] / "fixtures" / "attack" / "index_subset.json"
GROUND_TRUTH = {
    "label",
    "label_raw",
    "family",
    "is_attack",
    "split_chrono",
    "split_heldout",
    "features",
}


@pytest.fixture(scope="module")
def registry(
    event_engine: Engine, fixture_service: DetectorService, tmp_path_factory: pytest.TempPathFactory
) -> ToolRegistry:
    from secops.tools.nvd import NvdClient

    reg = build_registry(
        engine=event_engine,
        attack_index_path=SUBSET,
        nvd_client=NvdClient(fetcher=FixtureFetcher(), cache_dir=tmp_path_factory.mktemp("nvd")),
        include_detector=False,
    )
    for spec in DetectorTool(fixture_service, event_engine).specs():
        reg.register(spec)
    return reg


def _property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key, value in schema.get("properties", {}).items():
        names.add(key)
        if isinstance(value, dict):
            names |= _property_names(value)
    for d in schema.get("$defs", {}).values():
        names |= _property_names(d)
    return names


def _int_fields(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for key, value in schema.get("properties", {}).items():
        if isinstance(value, dict):
            types = [value.get("type")] + [
                a.get("type") for a in value.get("anyOf", []) if isinstance(a, dict)
            ]
            if "integer" in types:
                out.append((key, value))
    return out


def test_all_expected_tools_registered(registry: ToolRegistry) -> None:
    assert set(registry.names()) == set(EXPECTED_TOOLS)
    for spec in registry.all():
        assert spec.read_only is True
        assert spec.name == spec.name.lower() and " " not in spec.name
        assert "cannot" in spec.description.lower(), spec.name


def test_schemas_generate_and_bound_their_integers(registry: ToolRegistry) -> None:
    for spec in registry.all():
        inp, out = spec.input_schema(), spec.output_schema()
        assert inp["type"] == "object" and out["type"] == "object"
        assert inp.get("additionalProperties") is False, f"{spec.name} input must forbid extras"
        for key, field in _int_fields(inp):
            bounded = any(k in field for k in ("maximum", "exclusiveMaximum")) or any(
                k in a for a in field.get("anyOf", []) if isinstance(a, dict) for k in ("maximum",)
            )
            assert bounded or key == "event_id", f"{spec.name}.{key} is an unbounded integer"


def test_no_tool_output_exposes_ground_truth(registry: ToolRegistry) -> None:
    for spec in registry.all():
        names = _property_names(spec.output_schema())
        assert not (names & GROUND_TRUTH), f"{spec.name} exposes {names & GROUND_TRUTH}"
        assert "source" in names, f"{spec.name} output must name its source"


def test_invoke_validates_raw_arguments(registry: ToolRegistry) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        registry.get("search_events").invoke({"start": "2017-07-04T12:00:00+00:00"})
    res = registry.get("get_asset").invoke({"ip": "192.168.10.50"})
    assert res.model_dump()["status"] == "found"  # type: ignore[union-attr]
