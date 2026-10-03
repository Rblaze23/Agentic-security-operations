from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import OperationalError

from secops.db.events_loader import event_id_for, load_events, unpack_features
from secops.db.models import Event, LoadRun
from secops.db.session import make_engine, session_scope
from secops.detection.features import FEATURE_SPEC_V1


def test_loader_inserts_every_fixture_row_and_round_trips_features(
    fixture_parquet: Path, migrated_engine: Engine
) -> None:
    report = load_events(fixture_parquet, migrated_engine, FEATURE_SPEC_V1, chunk_size=100)
    df = pd.read_parquet(fixture_parquet)
    assert report.rows == len(df) and report.seconds >= 0 and report.db_bytes > 0
    with session_scope(migrated_engine) as s:
        assert s.scalar(select(func.count()).select_from(Event)) == len(df)
        row = df.iloc[7]
        ev = s.get(Event, event_id_for(str(row["day"]), int(row["id"])))
        assert ev is not None
        assert ev.source_row_id == int(row["id"]) and ev.day == str(row["day"])
        assert ev.source_ip == str(row["Src IP"]) and ev.destination_port == int(row["Dst Port"])
        assert ev.ts_us == int(row["Timestamp"].timestamp() * 1_000_000)
        assert ev.label_raw == str(row["label_raw"]) and ev.is_attack == int(row["is_attack"])
        assert ev.feature_spec_version == "v1-noport"
        vec = unpack_features(ev.features)
        expected = FEATURE_SPEC_V1.to_matrix(df.iloc[[7]])[0]
        assert vec.dtype == np.float32 and np.array_equal(vec, expected, equal_nan=True)
        assert ev.fwd_packets == int(row["Total Fwd Packet"])
        assert ev.syn_count == int(row["SYN Flag Count"])


def test_loader_is_idempotent_and_records_load_runs(
    fixture_parquet: Path, migrated_engine: Engine
) -> None:
    load_events(fixture_parquet, migrated_engine, FEATURE_SPEC_V1)
    load_events(fixture_parquet, migrated_engine, FEATURE_SPEC_V1)
    with session_scope(migrated_engine) as s:
        n = s.scalar(select(func.count()).select_from(Event))
        runs = [
            (r.rows, r.manifest_digest) for r in s.scalars(select(LoadRun).order_by(LoadRun.id))
        ]
    assert n == len(pd.read_parquet(fixture_parquet))
    assert len(runs) == 2 and runs[-1][0] == n and runs[-1][1]


def test_read_only_engine_refuses_writes(
    fixture_parquet: Path, migrated_engine: Engine, db_url: str
) -> None:
    load_events(fixture_parquet, migrated_engine, FEATURE_SPEC_V1)
    ro = make_engine(db_url, read_only=True)
    with ro.connect() as conn:
        assert conn.scalar(text("select count(*) from events")) > 0
        with pytest.raises(OperationalError):
            conn.execute(text("delete from events"))
