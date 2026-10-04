"""PostgreSQL integration (`pytest tests/integration -m postgres`): migrations from zero and the
investigation repository on a real server. URL from SECOPS_TEST_DATABASE_URL."""

from __future__ import annotations

import os
from typing import Any

import pytest
from sqlalchemy import inspect, text

from secops.agent.repository import InvestigationRepository
from secops.db.session import make_engine, upgrade_to_head
from secops.tools.registry import ToolRegistry

pytestmark = pytest.mark.postgres
URL = os.environ.get(
    "SECOPS_TEST_DATABASE_URL", "postgresql+psycopg://secops:secops@localhost:5432/secops_test"
)


@pytest.fixture(scope="module")
def pg_url() -> str:
    engine = make_engine(URL)
    with engine.begin() as conn:  # start from zero every module run
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    upgrade_to_head(URL)
    upgrade_to_head(URL)  # idempotent
    return URL


def test_postgres_migrations_from_zero(pg_url: str) -> None:
    insp = inspect(make_engine(pg_url))
    tables = set(insp.get_table_names())
    assert {"events", "load_runs", "investigations", "tool_calls", "model_predictions"} <= tables
    assert ("source_ip", "ts_us") in {tuple(i["column_names"]) for i in insp.get_indexes("events")}
    fks = insp.get_foreign_keys("tool_calls")
    assert fks and fks[0]["referred_table"] == "investigations"


def test_repository_round_trip_on_postgres(pg_url: str, registry: ToolRegistry, flows: Any) -> None:
    from tests.unit.agent.test_repository import _run

    result = _run(registry, flows)
    repo = InvestigationRepository(make_engine(pg_url))
    repo.save(result)
    stored = repo.get(result.investigation_id)
    assert stored is not None and stored.report == result.report
    assert [c.tool for c in stored.tool_calls] == [c.tool for c in result.tool_calls]
    assert stored.created_at.tzinfo is not None
    assert [r.investigation_id for r in repo.recent()][0] == result.investigation_id
