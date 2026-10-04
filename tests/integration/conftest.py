"""Integration tests reuse the unit fixtures (fixture store, registries)."""

from tests.unit.agent.conftest import full_registry, registry  # noqa: F401
from tests.unit.conftest import event_engine, fixture_parquet, fixture_service, flows  # noqa: F401
