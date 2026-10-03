"""Download the enterprise ATT&CK STIX bundle and build the technique index used by the tool.

Usage:
  uv run python scripts/fetch_attack.py            # index -> $SECOPS_DATA_DIR/attack/index.json
  uv run python scripts/fetch_attack.py --subset T1110,T1046 --out <fixture index path>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

from secops.config import get_settings
from secops.tools.attack import ATTACK_BUNDLE_URL, build_index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle", type=Path, default=None, help="existing bundle JSON (skips download)"
    )
    ap.add_argument("--out", type=Path, default=None, help="index path (default: data dir)")
    ap.add_argument("--subset", default=None, help="comma-separated technique ids to keep")
    args = ap.parse_args()

    attack_dir = get_settings().data_dir / "attack"
    bundle_path = args.bundle or attack_dir / "enterprise-attack.json"
    if not bundle_path.exists():
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(ATTACK_BUNDLE_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with bundle_path.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"downloaded {ATTACK_BUNDLE_URL} -> {bundle_path}")
    bundle = json.loads(bundle_path.read_text())
    fetched = datetime.now(UTC).date().isoformat()
    index = build_index(bundle, source_url=ATTACK_BUNDLE_URL, fetched=fetched)
    if args.subset:
        index = index.subset(args.subset.split(","))
    out = args.out or attack_dir / "index.json"
    index.save(out)
    print(f"ATT&CK v{index.version}: {len(index.techniques)} techniques -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
