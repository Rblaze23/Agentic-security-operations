"""Local data for the dashboard: run files, the golden set, the recorded scenarios and the
README tables. Everything here works without the API; nothing is typed by hand."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "evaluation" / "runs"
GOLDEN = ROOT / "evaluation" / "golden" / "v1.json"
SCENARIOS_DIR = ROOT / "tests" / "fixtures" / "llm"
README = ROOT / "README.md"
FIGURES = ROOT / "docs" / "figures"

HEADLINE_RUNS = [
    ("Rule-based baseline (no model)", "baseline-rule-based"),
    ("Agent with the Sonnet critic", "agent-v1-k1"),
    ("Agent, rules-only critic (default)", "agent-nocritic-full"),
]
DEFAULT_RUN = "agent-nocritic-full"


def load_run(run_id: str) -> dict[str, Any] | None:
    path = RUNS_DIR / f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def list_runs() -> list[dict[str, Any]]:
    out = []
    for path in sorted(RUNS_DIR.glob("*.json")) if RUNS_DIR.exists() else []:
        try:
            d = json.loads(path.read_text())
        except Exception:  # noqa: S112 - a half-written file must not break the page
            continue
        if d.get("metrics"):
            out.append(d)
    return out


def golden_cases() -> dict[str, dict[str, Any]]:
    if not GOLDEN.exists():
        return {}
    return {c["case_id"]: c for c in json.loads(GOLDEN.read_text())["cases"]}


def mean(metrics: dict[str, Any], key: str) -> float:
    v = metrics[key]
    return float(v["mean"]) if isinstance(v, dict) else float(v)


def headline_rows() -> list[dict[str, Any]]:
    rows = []
    for label, run_id in HEADLINE_RUNS:
        r = load_run(run_id)
        if not r or not r.get("metrics"):
            continue
        m = r["metrics"]
        rows.append(
            {
                "configuration": label,
                "verdict accuracy": round(mean(m, "verdict_accuracy"), 3),
                "composite": round(m["composite"], 3),
                "evidence recall": round(mean(m, "evidence_recall"), 3),
                "grounding": round(mean(m, "grounding_rate"), 3),
                "unsupported refs": m["unsupported_refs_total"],
                "latency p50 (s)": round(m["latency_p50_ms"] / 1000, 1),
                "cost / case (USD)": round(m["cost_per_case_usd"], 3),
                "cases": m["cases"],
                "run_id": run_id,
            }
        )
    return rows


def per_case_rows(run_ids: list[str]) -> list[dict[str, Any]]:
    """One row per golden case with the verdict each run gave it."""
    golden = golden_cases()
    verdicts: dict[str, dict[str, str]] = {}
    costs: dict[str, dict[str, float]] = {}
    for run_id in run_ids:
        r = load_run(run_id)
        if not r:
            continue
        for res in r["results"]:
            if res["repeat"] != 0:
                continue
            verdicts.setdefault(res["case_id"], {})[run_id] = res["score"]["verdict"]
            costs.setdefault(res["case_id"], {})[run_id] = res["score"]["cost_usd"]
    rows = []
    for case_id, case in golden.items():
        row: dict[str, Any] = {
            "case": case_id,
            "kind": case["kind"],
            "label (ground truth)": case["label"],
            "expected verdict": case["expected_verdict"],
            "expected severity": case["expected_severity"],
            "detector p": round(case["detector_probability"], 4),
        }
        for run_id in run_ids:
            v = verdicts.get(case_id, {}).get(run_id)
            row[run_id] = ("✅ " if v == case["expected_verdict"] else "❌ ") + v if v else "-"
        rows.append(row)
    return rows


def investigations_from_runs(run_id: str) -> list[dict[str, Any]]:
    """Recorded investigations (reports included) from one evaluation run."""
    r = load_run(run_id)
    if not r:
        return []
    golden = golden_cases()
    out = []
    for res in r["results"]:
        if res["repeat"] != 0:
            continue
        rep = res["report"]
        case = golden.get(res["case_id"], {})
        out.append(
            {
                "source": f"run {run_id}",
                "investigation_id": f"{run_id}:{res['case_id']}",
                "case_id": res["case_id"],
                "kind": case.get("kind", "-"),
                "ground_truth": case.get("label", "-"),
                "expected_verdict": case.get("expected_verdict", "-"),
                "verdict": rep["verdict"],
                "severity": rep["severity"],
                "family": rep.get("attack_family"),
                "confidence": rep["confidence"],
                "cost_usd": res["score"]["cost_usd"],
                "latency_s": round(res["score"]["latency_ms"] / 1000, 1),
                "report": rep,
                "errors": res.get("errors", []),
            }
        )
    return out


def scenario_investigations() -> list[dict[str, Any]]:
    """The four recorded Phase 4 scenarios (replayable in CI)."""
    out = []
    for d in sorted(SCENARIOS_DIR.iterdir()) if SCENARIOS_DIR.exists() else []:
        if not (d / "report.json").exists():
            continue
        rep = json.loads((d / "report.json").read_text())
        res = json.loads((d / "result.json").read_text())
        out.append(
            {
                "source": "Phase 4 scenario",
                "investigation_id": f"scenario:{d.name}",
                "case_id": d.name,
                "kind": "scenario",
                "ground_truth": "-",
                "expected_verdict": "-",
                "verdict": rep["verdict"],
                "severity": rep["severity"],
                "family": rep.get("attack_family"),
                "confidence": rep["confidence"],
                "cost_usd": res["cost_usd"],
                "latency_s": round(res["wall_s"], 1),
                "report": rep,
                "errors": res.get("errors", []),
            }
        )
    return out


def readme_tables(section_start: str, section_end: str) -> list[list[list[str]]]:
    """Markdown tables between two README headings, as lists of rows (cells unformatted)."""
    text = README.read_text()
    i, j = text.find(section_start), text.find(section_end)
    if i < 0 or j < 0:
        return []
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text[i:j].splitlines():
        if line.startswith("|"):
            cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def readme_section(section_start: str, section_end: str) -> str:
    text = README.read_text()
    i, j = text.find(section_start), text.find(section_end)
    return text[i:j] if i >= 0 and j >= 0 else ""


def table_to_rows(table: list[list[str]]) -> list[dict[str, str]]:
    header, *body = table
    return [dict(zip(header, row, strict=False)) for row in body]


def figure(name: str) -> Path | None:
    p = FIGURES / f"{name}.png"
    return p if p.exists() else None


def interview_answer(question_prefix: str) -> str:
    """One answer from docs/interview-notes.md, by the start of its bold question."""
    notes = (ROOT / "docs" / "interview-notes.md").read_text()
    m = re.search(
        rf"\*\*{re.escape(question_prefix)}[^\n]*\*\*\n(.*?)(?=\n\*\*|\n## |\Z)", notes, re.S
    )
    return m.group(1).strip() if m else ""
