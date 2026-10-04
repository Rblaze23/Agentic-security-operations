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


def test_upgrade_creates_investigations_and_tool_calls(migrated_engine: Engine) -> None:
    insp = inspect(migrated_engine)
    assert {"investigations", "tool_calls"} <= set(insp.get_table_names())
    cols = {c["name"] for c in insp.get_columns("investigations")}
    assert {
        "investigation_id",
        "alert_id",
        "event_id",
        "verdict",
        "severity",
        "cost_usd",
        "prompt_version",
        "report_json",
        "created_at",
    } <= cols
    assert {c["name"] for c in insp.get_columns("tool_calls")} >= {
        "investigation_id",
        "evidence_id",
        "tool",
        "arguments_json",
        "status",
        "latency_ms",
    }
    fks = insp.get_foreign_keys("tool_calls")
    assert fks and fks[0]["referred_table"] == "investigations"


def test_upgrade_creates_model_predictions(migrated_engine: Engine) -> None:
    insp = inspect(migrated_engine)
    assert "model_predictions" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("model_predictions")}
    assert {
        "prediction_id",
        "event_id",
        "attack_probability",
        "is_alert",
        "model_name",
        "created_at",
    } <= cols
    assert ("created_at",) in {
        tuple(i["column_names"]) for i in insp.get_indexes("model_predictions")
    }
