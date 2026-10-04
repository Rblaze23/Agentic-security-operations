"""Before/after comparison of two runs with regression gates, and markdown rendering."""

from __future__ import annotations

from pydantic import BaseModel

from secops.evaluation.metrics import MetricSummary, RunMetrics
from secops.evaluation.runner import RunRecord


class Gates(BaseModel):
    composite_max_drop: float = 0.02
    grounding_max_drop: float = 0.02
    cost_max_rise: float = 0.25
    unsupported_refs_max: int = 0
    require_same_cases: bool = True


class GateResult(BaseModel):
    name: str
    baseline: float
    candidate: float
    limit: float
    passed: bool
    detail: str


class ComparisonReport(BaseModel):
    baseline_run: str
    candidate_run: str
    gates: list[GateResult]
    passed: bool
    table: list[tuple[str, str, str, str]]


ROWS: list[tuple[str, str, str]] = [  # (label, RunMetrics attribute, kind)
    ("Composite score", "composite", "float"),
    ("Verdict accuracy", "verdict_accuracy", "summary"),
    ("Family agreement", "family_agreement", "summary"),
    ("Severity exact", "severity_exact", "summary"),
    ("Severity within one", "severity_within_one", "summary"),
    ("Evidence recall", "evidence_recall", "summary"),
    ("Evidence precision", "evidence_precision", "summary"),
    ("Grounding rate", "grounding_rate", "summary"),
    ("Judge-supported rate", "judge_supported_rate", "optional"),
    ("Techniques found", "techniques_ok", "summary"),
    ("CVEs found", "cves_ok", "summary"),
    ("Adversarial resisted", "adversarial_resisted", "optional"),
    ("Unsupported refs (total)", "unsupported_refs_total", "int"),
    ("Tool calls / case", "tool_calls_mean", "float"),
    ("Unnecessary calls / case", "unnecessary_tool_calls_mean", "float"),
    ("Loop rate (two critic rejections)", "loop_rate", "float"),
    ("Budget exhausted rate", "budget_exhausted_rate", "float"),
    ("Failure rate", "failure_rate", "float"),
    ("Latency p50 (s)", "latency_p50_ms", "ms"),
    ("Latency p95 (s)", "latency_p95_ms", "ms"),
    ("Cost / case (USD)", "cost_per_case_usd", "usd"),
    ("Cost total (USD)", "cost_total_usd", "usd"),
]


def _fmt(m: RunMetrics, attr: str, kind: str) -> str:
    v = getattr(m, attr)
    if kind == "summary" or kind == "optional":
        if v is None:
            return "n/a"
        s: MetricSummary = v
        return f"{s.mean:.3f}" + (f" ± {s.std:.3f}" if m.repeats > 1 else "")
    if kind == "int":
        return str(v)
    if kind == "ms":
        return f"{v / 1000:.1f}"
    if kind == "usd":
        return f"${v:.3f}"
    return f"{v:.3f}"


def _value(m: RunMetrics, attr: str, kind: str = "") -> float | None:
    v = getattr(m, attr)
    if v is None:
        return None
    out = float(v.mean) if isinstance(v, MetricSummary) else float(v)
    return out / 1000 if kind == "ms" else out


def compare_runs(
    baseline: RunRecord, candidate: RunRecord, gates: Gates | None = None
) -> ComparisonReport:
    gates = gates or Gates()
    assert baseline.metrics is not None and candidate.metrics is not None
    b, c = baseline.metrics, candidate.metrics
    results: list[GateResult] = []

    same_golden = baseline.golden_version == candidate.golden_version
    results.append(
        GateResult(
            name="same_golden_set",
            baseline=0.0,
            candidate=0.0,
            limit=0.0,
            passed=same_golden,
            detail=f"{baseline.golden_version} vs {candidate.golden_version}",
        )
    )
    if gates.require_same_cases:
        bc = {r.case_id for r in baseline.results}
        cc = {r.case_id for r in candidate.results}
        same = bc == cc
        results.append(
            GateResult(
                name="same_cases",
                baseline=len(bc),
                candidate=len(cc),
                limit=0,
                passed=same,
                detail="identical case sets"
                if same
                else f"missing {sorted(bc - cc)[:5]} extra {sorted(cc - bc)[:5]}",
            )
        )
    drop = b.composite - c.composite
    results.append(
        GateResult(
            name="composite_drop",
            baseline=b.composite,
            candidate=c.composite,
            limit=gates.composite_max_drop,
            passed=drop <= gates.composite_max_drop,
            detail=f"drop {drop:+.3f} (max {gates.composite_max_drop})",
        )
    )
    gdrop = b.grounding_rate.mean - c.grounding_rate.mean
    results.append(
        GateResult(
            name="grounding_drop",
            baseline=b.grounding_rate.mean,
            candidate=c.grounding_rate.mean,
            limit=gates.grounding_max_drop,
            passed=gdrop <= gates.grounding_max_drop,
            detail=f"drop {gdrop:+.3f} (max {gates.grounding_max_drop})",
        )
    )
    if b.cost_per_case_usd > 0:
        rise = (c.cost_per_case_usd - b.cost_per_case_usd) / b.cost_per_case_usd
        results.append(
            GateResult(
                name="cost_rise",
                baseline=b.cost_per_case_usd,
                candidate=c.cost_per_case_usd,
                limit=gates.cost_max_rise,
                passed=rise <= gates.cost_max_rise,
                detail=f"rise {rise:+.0%} (max {gates.cost_max_rise:.0%})",
            )
        )
    else:
        results.append(
            GateResult(
                name="cost_rise",
                baseline=0.0,
                candidate=c.cost_per_case_usd,
                limit=gates.cost_max_rise,
                passed=True,
                detail="baseline cost is 0 (rule-based); gate skipped",
            )
        )
    results.append(
        GateResult(
            name="unsupported_refs",
            baseline=b.unsupported_refs_total,
            candidate=c.unsupported_refs_total,
            limit=gates.unsupported_refs_max,
            passed=c.unsupported_refs_total <= gates.unsupported_refs_max,
            detail=f"candidate has {c.unsupported_refs_total} (max {gates.unsupported_refs_max})",
        )
    )
    table: list[tuple[str, str, str, str]] = []
    for label, attr, kind in ROWS:
        bv, cv = _value(b, attr, kind), _value(c, attr, kind)
        delta = "" if bv is None or cv is None else f"{cv - bv:+.3f}"
        table.append((label, _fmt(b, attr, kind), _fmt(c, attr, kind), delta))
    return ComparisonReport(
        baseline_run=baseline.config.run_id,
        candidate_run=candidate.config.run_id,
        gates=results,
        passed=all(g.passed for g in results),
        table=table,
    )


def render_metrics(run: RunRecord) -> str:
    m = run.metrics
    assert m is not None
    lines = [
        f"### Run `{run.config.run_id}` ({run.config.investigator}, {m.cases} cases x "
        f"{m.repeats} repeats, prompt {run.prompt_version}, "
        f"models {run.models['investigator']} / {run.models['critic']})",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {label} | {_fmt(m, attr, kind)} |" for label, attr, kind in ROWS]
    if m.flaky_cases:
        lines.append(f"| Flaky cases | {', '.join(m.flaky_cases)} |")
    lines += [
        "",
        "| Kind | Cases | Verdict accuracy | Grounding | Evidence recall | Cost / case |",
        "|---|---|---|---|---|---|",
    ]
    for kind, v in m.by_kind.items():
        lines.append(
            f"| {kind} | {int(v['cases'])} | {v['verdict_accuracy']:.3f} | "
            f"{v['grounding_rate']:.3f} | {v['evidence_recall']:.3f} | "
            f"${v['cost_per_case_usd']:.3f} |"
        )
    return "\n".join(lines)


def render_comparison(c: ComparisonReport) -> str:
    lines = [
        f"### `{c.baseline_run}` (baseline) vs `{c.candidate_run}` (candidate)",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {d} | {e} |" for a, b, d, e in c.table]
    lines.append("")
    for g in c.gates:
        lines.append(f"- {'PASS' if g.passed else 'FAIL'} `{g.name}`: {g.detail}")
    lines.append("")
    lines.append(f"**Result: {'PASS' if c.passed else 'FAIL'}**")
    return "\n".join(lines)
