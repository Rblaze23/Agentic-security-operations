from sqlalchemy import Engine, inspect

from secops.db.session import upgrade_to_head


def test_upgrade_creates_events_and_load_runs(migrated_engine: Engine) -> None:
    insp = inspect(migrated_engine)
    assert {"events", "load_runs", "alembic_version"} <= set(insp.get_table_names())
    cols = {c["name"] for c in insp.get_columns("events")}
    expected = {
        "event_id",
        "source_row_id",
        "day",
        "ts_us",
        "source_ip",
        "source_port",
        "destination_ip",
        "destination_port",
        "protocol",
        "flow_duration_us",
        "fwd_packets",
        "bwd_packets",
        "fwd_bytes",
        "bwd_bytes",
        "syn_count",
        "fin_count",
        "rst_count",
        "ack_count",
        "feature_spec_version",
        "features",
        "label_raw",
        "label",
        "family",
        "is_attack",
        "split_chrono",
        "split_heldout",
    }
    assert expected <= cols, expected - cols
    indexed = {tuple(i["column_names"]) for i in insp.get_indexes("events")}
    assert ("source_ip", "ts_us") in indexed
    assert ("destination_ip", "ts_us") in indexed
    assert ("ts_us",) in indexed
    assert ("destination_port",) in indexed


def test_upgrade_is_idempotent(db_url: str) -> None:
    upgrade_to_head(db_url)
    upgrade_to_head(db_url)  # second run is a no-op, not an error
