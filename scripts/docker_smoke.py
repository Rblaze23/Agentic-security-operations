"""Smoke-test a built secops-api image: build fixture bundles, run the container, call the API.

Usage: uv run python scripts/docker_smoke.py <image> [--docker docker]
Exit code 0 only if /health is ok and /predict answers a fixture flow with a valid prediction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `tests` importable

from tests.fixtures.make_bundle import FIXTURE_DIR, make_fixture_bundles  # noqa: E402

from secops.data.clean import AttemptedPolicy, clean  # noqa: E402
from secops.data.ingest import read_all  # noqa: E402
from secops.data.schema import FEATURE_COLS  # noqa: E402

API_KEY = "smoke-test-key"
PORT = 18000


def _http(
    method: str, path: str, body: dict[str, object] | None = None
) -> tuple[int, dict[str, object]]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310  local container
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--docker", default="docker")
    ap.add_argument(
        "--bundles-dir",
        default=None,
        help="where to write the fixture bundles (default: a temp dir); use a path the Docker "
        "engine can bind-mount when the CLI runs on another OS",
    )
    ap.add_argument(
        "--mount-path",
        default=None,
        help="how the Docker engine should spell --bundles-dir in -v (default: same path)",
    )
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="secops-smoke-") as tmp:
        bundles = Path(args.bundles_dir) if args.bundles_dir else Path(tmp) / "models"
        make_fixture_bundles(bundles)
        mount = args.mount_path or str(bundles)
        name = f"secops-smoke-{int(time.time())}"
        run = [
            args.docker,
            "run",
            "--rm",
            "-d",
            "--name",
            name,
            "-p",
            f"{PORT}:8000",
            "-e",
            f"SECOPS_API_KEYS={API_KEY}",
            "-v",
            f"{mount}:/models:ro",
            args.image,
        ]
        subprocess.run(run, check=True)  # noqa: S603
        try:
            status: int = 0
            body: dict[str, object] = {}
            for _ in range(60):
                time.sleep(1)
                try:
                    status, body = _http("GET", "/health")
                except (urllib.error.URLError, ConnectionError, OSError):
                    continue
                if status == 200 and body.get("status") == "ok":
                    break
            else:
                print("container never became healthy:", status, body, file=sys.stderr)
                subprocess.run([args.docker, "logs", name], check=False)  # noqa: S603
                return 1
            df, _ = clean(read_all(FIXTURE_DIR, subdir=""), AttemptedPolicy.RELABEL_BENIGN)
            row = df.iloc[0]
            payload: dict[str, object] = {
                "event_id": "smoke-1",
                "metadata": {"source_ip": str(row["Src IP"]), "destination_ip": str(row["Dst IP"])},
                "features": {
                    c: (None if row[c] != row[c] else float(row[c])) for c in FEATURE_COLS
                },
            }
            status, body = _http("POST", "/predict", payload)
            if status != 200:
                print("predict failed:", status, body, file=sys.stderr)
                return 1
            pred = body["prediction"]
            assert 0.0 <= pred["attack_probability"] <= 1.0 and len(pred["top_contributions"]) == 5  # type: ignore[index]
            status, _ = _http("POST", "/predict", {"features": {"x": 1}})
            assert status == 422, status
            print(json.dumps({"smoke": "ok", "image": args.image, "prediction": pred}, indent=2))
            return 0
        finally:
            subprocess.run([args.docker, "rm", "-f", name], check=False, capture_output=True)  # noqa: S603


if __name__ == "__main__":
    sys.exit(main())
