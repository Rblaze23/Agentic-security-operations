"""Figures are generated from run files, never from thin air."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_figures.py"


@pytest.mark.skipif(
    not (ROOT / "evaluation" / "runs" / "agent-v1-k1.json").exists(), reason="no agent run file"
)
def test_make_figures_writes_non_trivial_pngs() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True)
    assert "agent_vs_baseline.png" in out.stdout
    for name in ("agent_vs_baseline", "verdict_by_kind", "cost_latency", "detector_pr_auc"):
        png = ROOT / "docs" / "figures" / f"{name}.png"
        assert png.exists() and png.stat().st_size > 10_000, name


def test_make_figures_requires_inputs() -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--agent", "does-not-exist"], capture_output=True, text=True
    )
    assert out.returncode != 0 and "does-not-exist" in (out.stderr + out.stdout)
