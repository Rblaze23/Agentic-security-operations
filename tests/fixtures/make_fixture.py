"""Build a <=1,000-row fixture from the real improved CSVs: every label present, time order kept.

Usage: SECOPS_DATA_DIR=~/data/secops uv run python tests/fixtures/make_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from secops.config import get_settings
from secops.data.manifest import csv_path
from secops.data.schema import DAY_ORDER, LABEL_COL

OUT = Path(__file__).parent / "mini_cicids"
PER_LABEL_CAP = 30
BENIGN_PER_DAY = 60
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = get_settings().raw_dir
    for day in DAY_ORDER:
        df = pd.read_csv(csv_path(raw, day), low_memory=False)
        parts = []
        for label, g in df.groupby(LABEL_COL, sort=False):
            cap = BENIGN_PER_DAY if label == "BENIGN" else PER_LABEL_CAP
            take = g if len(g) <= cap else g.iloc[np.sort(rng.choice(len(g), cap, replace=False))]
            parts.append(take)
        out = pd.concat(parts).sort_values("id")
        out.to_csv(OUT / f"{day}.csv", index=False)
        print(day, len(out))


if __name__ == "__main__":
    main()
