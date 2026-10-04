"""Figures for the README and the dashboard, generated only from measured files.

    uv run python scripts/make_figures.py [--agent ID] [--baseline ID] [--k3 ID]

Every figure names its source run ids; a missing input raises FileNotFoundError, never an
empty chart."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evaluation" / "runs"
OUT = ROOT / "docs" / "figures"
METRICS = [
    ("verdict_accuracy", "Verdict accuracy"),
    ("family_agreement", "Family agreement"),
    ("severity_within_one", "Severity ±1"),
    ("evidence_recall", "Evidence recall"),
    ("grounding_rate", "Grounding"),
]


def _load(run_id: str) -> dict:
    path = RUNS / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"run file missing: {path}")
    data = json.loads(path.read_text())
    if not data.get("metrics"):
        raise FileNotFoundError(f"run {run_id} has no metrics (unfinished): {path}")
    return data


def _mean(m: dict, key: str) -> float:
    v = m[key]
    return float(v["mean"]) if isinstance(v, dict) else float(v)


def fig_agent_vs_baseline(agent: dict, baseline: dict) -> Path:
    am, bm = agent["metrics"], baseline["metrics"]
    labels = [lbl for _, lbl in METRICS] + ["Composite"]
    a = [_mean(am, k) for k, _ in METRICS] + [am["composite"]]
    b = [_mean(bm, k) for k, _ in METRICS] + [bm["composite"]]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - 0.2 for i in x], b, 0.4, label=f"rule-based ({baseline['config']['run_id']})")
    ax.bar([i + 0.2 for i in x], a, 0.4, label=f"agent ({agent['config']['run_id']})")
    ax.set_xticks(list(x), labels, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (0–1)")
    ax.set_title(
        f"Agent vs rule-based baseline on golden set {agent['golden_version']} "
        f"({am['cases']} cases)"
    )
    ax.legend(loc="lower right")
    for i, (va, vb) in enumerate(zip(a, b, strict=True)):
        ax.text(i + 0.2, va + 0.01, f"{va:.2f}", ha="center", fontsize=8)
        ax.text(i - 0.2, vb + 0.01, f"{vb:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = OUT / "agent_vs_baseline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_verdict_by_kind(agent: dict, baseline: dict) -> Path:
    kinds = ["attack", "benign_fp", "adversarial"]
    a = [agent["metrics"]["by_kind"].get(k, {}).get("verdict_accuracy", 0.0) for k in kinds]
    b = [baseline["metrics"]["by_kind"].get(k, {}).get("verdict_accuracy", 0.0) for k in kinds]
    n = [int(agent["metrics"]["by_kind"].get(k, {}).get("cases", 0)) for k in kinds]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(kinds))
    ax.bar([i - 0.2 for i in x], b, 0.4, label="rule-based")
    ax.bar([i + 0.2 for i in x], a, 0.4, label="agent")
    ax.set_xticks(list(x), [f"{k}\n(n={c})" for k, c in zip(kinds, n, strict=True)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("verdict accuracy")
    ax.set_title("Verdict accuracy by case kind")
    ax.legend()
    fig.tight_layout()
    out = OUT / "verdict_by_kind.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_cost_latency(agent: dict, baseline: dict) -> Path:
    am, bm = agent["metrics"], baseline["metrics"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.bar(["rule-based", "agent"], [bm["cost_per_case_usd"], am["cost_per_case_usd"]])
    ax1.set_title("Cost per investigation (USD)")
    for i, v in enumerate([bm["cost_per_case_usd"], am["cost_per_case_usd"]]):
        ax1.text(i, v, f"${v:.3f}", ha="center", va="bottom", fontsize=9)
    agent_lat = [am["latency_p50_ms"] / 1000, am["latency_p95_ms"] / 1000]
    base_lat = [bm["latency_p50_ms"] / 1000, bm["latency_p95_ms"] / 1000]
    ax2.bar(["p50", "p95"], agent_lat, label="agent")
    ax2.bar(["p50 ", "p95 "], base_lat, label="rule-based")
    ax2.set_title("Latency (s)")
    ax2.legend()
    fig.suptitle(f"{agent['config']['run_id']} vs {baseline['config']['run_id']}", fontsize=9)
    fig.tight_layout()
    out = OUT / "cost_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_reliability(k3: dict) -> Path:
    per_case: dict[str, list[str]] = {}
    for r in k3["results"]:
        per_case.setdefault(r["case_id"], []).append(r["score"]["verdict"])
    cases = sorted(per_case)
    agreement = [len(set(v)) == 1 for v in (per_case[c] for c in cases)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(
        range(len(cases)),
        [1 if a else 0.35 for a in agreement],
        color=["#2a9d8f" if a else "#e76f51" for a in agreement],
    )
    ax.set_xticks(range(len(cases)), cases, rotation=60, ha="right", fontsize=7)
    ax.set_yticks([0.35, 1], ["verdicts differ", "same verdict"])
    ax.set_title(
        f"Verdict agreement across k={k3['metrics']['repeats']} repeats "
        f"({k3['config']['run_id']}): {sum(agreement)}/{len(cases)} stable"
    )
    fig.tight_layout()
    out = OUT / "reliability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_detector(evaluation_md: Path) -> Path:
    """PR-AUC per model from the Phase 1 chronological-split table in docs/evaluation.md."""
    text = evaluation_md.read_text()
    # run table: | run_name | model | weighting | val_pr_auc | test_pr_auc | ...
    found = re.findall(
        r"^\| (\w+_(?:none|balanced))\s*\| \w+\s*\| \w+\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|",
        text,
        re.M,
    )
    rows = [(m, float(pr)) for m, pr in found][:6]
    if not rows:
        raise FileNotFoundError("no detector table rows found in docs/evaluation.md")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([m for m, _ in rows], [v for _, v in rows])
    ax.set_xlim(0.95, 1.0)
    ax.set_xlabel("test PR-AUC (chronological split)")
    ax.set_title("Detector candidates, Phase 1 (docs/evaluation.md)")
    for i, (_, v) in enumerate(rows):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    out = OUT / "detector_pr_auc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent-v1-k1")
    ap.add_argument("--baseline", default="baseline-rule-based")
    ap.add_argument("--k3", default="agent-v1-k3")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    agent, baseline = _load(args.agent), _load(args.baseline)
    outs = [
        fig_agent_vs_baseline(agent, baseline),
        fig_verdict_by_kind(agent, baseline),
        fig_cost_latency(agent, baseline),
    ]
    if (RUNS / f"{args.k3}.json").exists():
        outs.append(fig_reliability(_load(args.k3)))
    outs.append(fig_detector(ROOT / "docs" / "evaluation.md"))
    for o in outs:
        print(f"{o.relative_to(ROOT)}  {o.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
