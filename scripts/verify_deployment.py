"""Verify a deployed detection API: health, model, one prediction, investigations state.

Usage: uv run python scripts/verify_deployment.py https://<service-url> <api-key>
Prints a markdown table that docs/deployment.md pastes verbatim (measured, not assumed)."""

from __future__ import annotations

import sys
import time

import pandas as pd
import requests
from tests.conftest import FIXTURE_DIR

from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all
from secops.data.schema import FEATURE_COLS


def _ms(t: float) -> str:
    return f"{(time.perf_counter() - t) * 1000:.0f} ms"


def main(url: str, key: str) -> int:
    url = url.rstrip("/")
    headers = {"X-API-Key": key}
    rows: list[tuple[str, str, str]] = []
    expected: list[set[str]] = []  # accepted statuses per row, in order

    t = time.perf_counter()
    r = requests.get(f"{url}/health", timeout=30)
    rows.append(("GET /health", str(r.status_code), f"{_ms(t)}, {r.json().get('status')}"))
    expected.append({"200"})

    t = time.perf_counter()
    r = requests.get(f"{url}/model", headers=headers, timeout=30)
    if r.ok:
        det = r.json()["detector"]
        detail = f"{det['model_name']} v{det['version']}"
    else:
        detail = r.text[:80]
    rows.append(("GET /model", str(r.status_code), f"{_ms(t)}, {detail}"))
    expected.append({"200"})

    df, _ = clean(read_all(FIXTURE_DIR, subdir=""), AttemptedPolicy.RELABEL_BENIGN)
    row = df[df["label"].astype(str) == "FTP-Patator"].iloc[0]
    body = {
        "metadata": {
            "source_ip": str(row["Src IP"]),
            "destination_ip": str(row["Dst IP"]),
            "destination_port": int(row["Dst Port"]),
            "protocol": int(row["Protocol"]),
        },
        "features": {c: (None if pd.isna(row[c]) else float(row[c])) for c in FEATURE_COLS},
    }
    t = time.perf_counter()
    r = requests.post(f"{url}/predict", json=body, headers=headers, timeout=60)
    if r.ok:
        pred = r.json()["prediction"]
        detail = f"p={pred['attack_probability']:.4f}, alert={pred['is_alert']}"
    else:
        detail = r.text[:80]
    rows.append(
        ("POST /predict (FTP-Patator fixture row)", str(r.status_code), f"{_ms(t)}, {detail}")
    )

    r = requests.get(f"{url}/investigations", headers=headers, timeout=30)
    inv = "disabled (503) on the public deployment" if r.status_code == 503 else r.text[:80]
    rows.append(("GET /investigations", str(r.status_code), inv))
    expected.append({"200", "503"})

    r = requests.get(f"{url}/model", timeout=30)
    rows.append(("GET /model without key", str(r.status_code), "must be 401"))
    expected.append({"401"})

    print("| Check | Status | Detail |")
    print("|---|---|---|")
    for a, b, c in rows:
        print(f"| {a} | {b} | {c} |")
    failures = [row[0] for row, ok in zip(rows, expected, strict=True) if row[1] not in ok]
    if failures:
        print(f"\nFAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
