"""End-to-end demo: start the API, submit the recorded FTP brute-force alert, print the triage.

    uv run python scripts/demo.py            # replay mode: no key, no model, no event store
    uv run python scripts/demo.py --live     # real model calls (needs ANTHROPIC_API_KEY, bundles)

Replay mode builds fixture bundles into a temp dir, serves the recorded ftp_bruteforce scenario
through the real HTTP endpoints and persists the investigation in a temp SQLite file."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PORT = 18010
KEY = "demo-key"


def _http(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - localhost only
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    from tests.fixtures.make_bundle import make_fixture_bundles

    with tempfile.TemporaryDirectory(prefix="secops-demo-") as tmp:
        env = dict(os.environ)
        env.update(
            SECOPS_API_KEYS=KEY,
            SECOPS_PORT=str(PORT),
            SECOPS_RATE_LIMIT_PER_MINUTE="0",
            SECOPS_DATABASE_URL=f"sqlite:///{Path(tmp) / 'demo.db'}",
            SECOPS_LOG_LEVEL="WARNING",
        )
        if args.live:
            env.setdefault("SECOPS_AGENT_MODE", "live")
            print(
                "live mode: one investigation costs about $0.20 at 2026-10-04 prices "
                "(Opus 5.5 + Sonnet 5.5)"
            )
        else:
            det, _fam = make_fixture_bundles(Path(tmp) / "models")
            env.update(
                SECOPS_MODEL_DIR=str(det.parent),
                SECOPS_AGENT_MODE="replay",
                SECOPS_AGENT_FIXTURE_ROOT=str(ROOT / "tests" / "fixtures" / "llm"),
                SECOPS_AGENT_SCENARIO="ftp_bruteforce",
                SECOPS_LLM_CRITIC="true",  # the scenario was recorded with the critic
            )
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "secops.api"],
            env=env,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(60):
                time.sleep(1)
                try:
                    status, body = _http("GET", "/health")
                    if status == 200 and body.get("status") in ("ok", "degraded"):
                        break
                except Exception:  # noqa: S112 - the API is still starting
                    continue
            else:
                print(
                    "API did not start:", proc.stderr.read().decode()[-2000:] if proc.stderr else ""
                )
                return 1
            alert = json.loads((ROOT / "tests/fixtures/llm/ftp_bruteforce/alert.json").read_text())
            print(
                f"alert {alert['alert_id']}: {alert['metadata']['source_ip']} -> "
                f"{alert['metadata']['destination_ip']}:{alert['metadata']['destination_port']}, "
                f"p={alert['prediction']['attack_probability']:.4f} "
                f"({alert['prediction']['predicted_family']})"
            )
            status, body = _http("POST", "/investigations", {"alert": alert})
            print(
                f"POST /investigations -> {status} {body.get('status')} "
                f"{body.get('investigation_id')}"
            )
            inv_id = body["investigation_id"]
            for _ in range(180):
                time.sleep(1)
                try:  # a synchronous background task blocks the server; retry while it runs
                    status, body = _http("GET", f"/investigations/{inv_id}")
                except (urllib.error.URLError, ConnectionError, http.client.HTTPException):
                    continue
                if body.get("status") in ("done", "failed"):
                    break
            print(
                f"GET /investigations/{inv_id} -> {body.get('status')}: "
                f"verdict {body.get('verdict')}, "
                f"severity {body.get('severity')}, cost ${body.get('cost_usd') or 0:.4f}"
            )
            if body.get("error"):
                print("error:", body["error"])
                return 1
            report = body["report"]
            print("\n" + report["summary"])
            print("\nfindings:")
            for f in report["findings"]:
                print(f"  [{f['kind']}] {f['statement'][:140]} {f['evidence_ids']}")
            print("\nsteps:")
            for s in report["investigation_steps"]:
                print("  " + s[:140])
            print(
                "\nuncertainties:",
                len(report["uncertainties"]),
                "| techniques:",
                [t["technique_id"] for t in report["attack_techniques"]],
            )
            return 0 if body.get("status") == "done" else 1
        except Exception as e:
            print(f"demo failed: {type(e).__name__}: {e}")
            proc.terminate()
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            print("--- API stderr (tail) ---\n" + err[-3000:])
            return 1
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
