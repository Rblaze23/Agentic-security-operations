"""Engine construction (optionally read-only), sessions, and programmatic migrations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def make_engine(url: str, read_only: bool = False) -> Engine:
    """Engine for `url`. With read_only=True every connection refuses writes.

    SQLite: `PRAGMA query_only = 1` on connect. PostgreSQL (Phase 6): use a SELECT-only role in
    the URL; the pragma is a no-op there and the role does the enforcement.
    """
    kwargs: dict[str, Any] = {"future": True}
    if url.startswith("postgresql"):
        # pre-ping drops stale pooled connections (Cloud SQL idles them); small pool per process
        kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=5)
    engine = create_engine(url, **kwargs)
    if read_only and url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _query_only(dbapi_conn: Any, _record: Any) -> None:
            dbapi_conn.execute("PRAGMA query_only = 1")

    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def upgrade_to_head(url: str) -> None:
    """Apply all migrations. Idempotent."""
    Path(url.removeprefix("sqlite:///")).parent.mkdir(
        parents=True, exist_ok=True
    ) if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:") else None
    command.upgrade(alembic_config(url), "head")
