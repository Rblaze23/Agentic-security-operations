"""The README has the brief's 16 sections in order and quotes only measured numbers."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()
SECTIONS = [
    "Project overview",
    "Problem statement",
    "Why an agentic architecture",
    "Architecture diagram",
    "ML pipeline",
    "Agent workflow",
    "Tool architecture",
    "Evaluation methodology",
    "Results",
    "Security considerations",
    "MLOps",
    "Local setup",
    "API documentation",
    "Deployment",
    "Limitations",
    "Future work",
]


def test_sixteen_sections_in_order() -> None:
    positions = []
    for i, title in enumerate(SECTIONS, start=1):
        m = re.search(rf"^## {i}\. {re.escape(title)}\s*$", README, re.M)
        assert m, f"missing section {i}. {title}"
        positions.append(m.start())
    assert positions == sorted(positions)


def test_no_placeholders() -> None:
    for bad in ("TODO", "lorem", "XXX"):
        assert bad not in README, bad


def test_results_numbers_come_from_run_files() -> None:
    runs = {
        name: json.loads((ROOT / "evaluation" / "runs" / f"{name}.json").read_text())["metrics"]
        for name in ("baseline-rule-based", "agent-v1-k1")
    }
    results = README[README.index("## 9. Results") : README.index("## 10. Security considerations")]
    row = next(line for line in results.splitlines() if line.startswith("| Verdict accuracy |"))
    for name in ("baseline-rule-based", "agent-v1-k1"):
        assert f"{runs[name]['verdict_accuracy']['mean']:.3f}" in row, name
    row = next(line for line in results.splitlines() if line.startswith("| Composite score |"))
    for name in ("baseline-rule-based", "agent-v1-k1"):
        assert f"{runs[name]['composite']:.3f}" in row, name
    cost = next(line for line in results.splitlines() if line.startswith("| Cost per case"))
    assert f"${runs['agent-v1-k1']['cost_per_case_usd']:.3f}" in cost
    assert f"${runs['agent-v1-k1']['cost_total_usd']:.2f}" in cost
    checks = {
        "| Family agreement |": lambda m: f"{m['family_agreement']['mean']:.3f}",
        "| Evidence recall / precision |": lambda m: f"{m['evidence_recall']['mean']:.3f}",
        "| Grounding rate": lambda m: f"{m['grounding_rate']['mean']:.3f}",
        "| Tool calls per case": lambda m: f"{m['tool_calls_mean']:.1f}",
        "| Latency p50 / p95 |": lambda m: f"{m['latency_p50_ms'] / 1000:.1f} s",
    }
    for prefix, fmt in checks.items():
        row = next(line for line in results.splitlines() if line.startswith(prefix))
        for name in ("baseline-rule-based", "agent-v1-k1"):
            assert fmt(runs[name]) in row, (prefix, name, fmt(runs[name]))
    cap = next(line for line in results.splitlines() if line.startswith("| Cap hit"))
    assert f"{runs['agent-v1-k1']['loop_rate'] * 100:.1f} %" in cap
    judge = next(line for line in results.splitlines() if line.startswith("| Judge-supported"))
    assert f"{runs['agent-v1-k1']['judge_supported_rate']['mean']:.3f}" in judge
