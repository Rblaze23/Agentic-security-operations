"""Registry of the tools available to the agent, and the factory that wires them to data."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import Engine

from secops.config import Settings, get_settings
from secops.db.session import make_engine
from secops.tools.attack import AttackIndex, AttackTools
from secops.tools.base import ToolSpec
from secops.tools.cve import CveTools
from secops.tools.enrichment import EnrichmentTools, load_seeds
from secops.tools.events import EventStoreTools
from secops.tools.nvd import NvdClient

log = logging.getLogger(__name__)

EXPECTED_TOOLS: tuple[str, ...] = (
    "search_events",
    "get_related_events",
    "get_asset",
    "enrich_ip",
    "lookup_attack_technique",
    "lookup_cve",
    "predict_attack",
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} registered twice")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)


def build_registry(
    settings: Settings | None = None,
    engine: Engine | None = None,
    attack_index_path: Path | None = None,
    nvd_client: NvdClient | None = None,
    include_detector: bool = True,
) -> ToolRegistry:
    """Wire every tool to its data source. Tests pass fixtures; production passes nothing.

    The event store is opened read-only. `include_detector=True` requires Phase 2's
    DetectorService and the exported bundles; it raises rather than silently omitting the tool.
    """
    settings = settings or get_settings()
    engine = engine or make_engine(settings.resolved_database_url(), read_only=True)
    index_path = attack_index_path or settings.data_dir / "attack" / "index.json"
    nvd = nvd_client or NvdClient(cache_dir=settings.data_dir / "cache" / "nvd")

    registry = ToolRegistry()
    for spec in EventStoreTools(engine).specs():
        registry.register(spec)
    for spec in EnrichmentTools(load_seeds()).specs():
        registry.register(spec)
    for spec in AttackTools(AttackIndex.load(index_path)).specs():
        registry.register(spec)
    for spec in CveTools(nvd).specs():
        registry.register(spec)
    if include_detector:
        from secops.tools.detector import detector_specs  # Phase 2 dependency, imported lazily

        for spec in detector_specs(settings, engine):
            registry.register(spec)
    log.info("tool registry built", extra={"extra": {"tools": registry.names()}})
    return registry
