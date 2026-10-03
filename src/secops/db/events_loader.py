"""Fill the events table from the processed Parquet. Idempotent: each load replaces the table."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import Engine, delete, insert

from secops.data.manifest import manifest_digest
from secops.data.schema import DAY_ORDER
from secops.db.models import Event, LoadRun
from secops.db.session import session_scope
from secops.detection.features import FeatureSpec

DERIVED: dict[str, str] = {
    "flow_duration_us": "Flow Duration",
    "fwd_packets": "Total Fwd Packet",
    "bwd_packets": "Total Bwd packets",
    "fwd_bytes": "Total Length of Fwd Packet",
    "bwd_bytes": "Total Length of Bwd Packet",
    "syn_count": "SYN Flag Count",
    "fin_count": "FIN Flag Count",
    "rst_count": "RST Flag Count",
    "ack_count": "ACK Flag Count",
}
META = ["id", "day", "Timestamp", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]
TRUTH = ["label_raw", "label", "family", "is_attack", "split_chrono", "split_heldout"]


@dataclass
class LoadReport:
    rows: int
    seconds: float
    db_bytes: int


DAY_STRIDE = 1_000_000  # every day file has fewer than a million rows


def event_id_for(day: str, source_row_id: int) -> int:
    """Globally unique event id: the dataset's `id` restarts at 1 in each day file."""
    return DAY_ORDER.index(day) * DAY_STRIDE + int(source_row_id)


def pack_features(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype=np.float32).tobytes()


def unpack_features(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603, S607
    except Exception:  # noqa: BLE001
        return "unknown"


def _rows_from_chunk(chunk: pd.DataFrame, spec: FeatureSpec) -> list[dict[str, object]]:
    X = spec.to_matrix(chunk)
    # Timestamp may be datetime64[us] or [ns] depending on the pandas version: normalise explicitly.
    ts_us = (
        chunk["Timestamp"].dt.tz_convert("UTC").astype("datetime64[us, UTC]").astype("int64")
    ).to_numpy()
    out: list[dict[str, object]] = []
    for i, (_, r) in enumerate(chunk.iterrows()):
        row: dict[str, object] = {
            "event_id": event_id_for(str(r["day"]), int(r["id"])),
            "source_row_id": int(r["id"]),
            "day": str(r["day"]),
            "ts_us": int(ts_us[i]),
            "source_ip": str(r["Src IP"]),
            "source_port": int(r["Src Port"]),
            "destination_ip": str(r["Dst IP"]),
            "destination_port": int(r["Dst Port"]),
            "protocol": int(r["Protocol"]),
            "feature_spec_version": spec.version,
            "features": pack_features(X[i]),
            "label_raw": str(r["label_raw"]),
            "label": str(r["label"]),
            "family": str(r["family"]),
            "is_attack": int(r["is_attack"]),
            "split_chrono": str(r["split_chrono"]),
            "split_heldout": str(r["split_heldout"]),
        }
        for col, feat in DERIVED.items():
            v = r[feat]
            row[col] = 0 if pd.isna(v) else int(v)
        out.append(row)
    return out


def load_events(
    parquet_path: Path,
    engine: Engine,
    feature_spec: FeatureSpec,
    chunk_size: int = 50_000,
    attempted_policy: str = "relabel_benign",
) -> LoadReport:
    start = time.perf_counter()
    columns = [*META, *feature_spec.names, *TRUTH]
    pf = pq.ParquetFile(parquet_path)
    total = 0
    with engine.begin() as conn:
        conn.execute(delete(Event))
        for batch in pf.iter_batches(batch_size=chunk_size, columns=columns):
            chunk = batch.to_pandas()
            rows = _rows_from_chunk(chunk, feature_spec)
            conn.execute(insert(Event), rows)
            total += len(rows)
    seconds = time.perf_counter() - start
    with session_scope(engine) as s:
        s.add(
            LoadRun(
                loaded_at=datetime.now(UTC).replace(tzinfo=None),
                source_path=str(parquet_path),
                attempted_policy=attempted_policy,
                feature_spec_version=feature_spec.version,
                manifest_digest=manifest_digest(),
                git_sha=_git_sha(),
                rows=total,
                seconds=seconds,
            )
        )
    db_bytes = 0
    url = str(engine.url)
    if url.startswith("sqlite:///") and ":memory:" not in url:
        db_bytes = Path(url.removeprefix("sqlite:///")).stat().st_size
    return LoadReport(rows=total, seconds=seconds, db_bytes=db_bytes)
