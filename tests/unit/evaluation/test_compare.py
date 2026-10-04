from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from secops.evaluation.compare import Gates, compare_runs, render_comparison, render_metrics
from secops.evaluation.metrics import aggregate
from secops.evaluation.runner import CaseResult, RunConfig, RunRecord
from tests.unit.evaluation.test_metrics import _case, _result, score_case


def _record(run_id: str, scores: list[Any]) -> RunRecord:
    results = [
        CaseResult(case_id=s.case_id, repeat=s.repeat, score=s, report=_result().report, errors=[])
        for s in scores
    ]
    return RunRecord(
        config=RunConfig(run_id=run_id, golden_path=Path("g.json"), recordings_dir=Path("r")),
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        prompt_version="p",
        models={"investigator": "m", "critic": "c"},
        golden_version="v1",
        results=results,
        metrics=aggregate(scores),
    )


def _scores(n: int, good: int, cost: float = 0.1) -> list[Any]:
    out = []
    for i in range(n):
        case = _case(case_id=f"c{i}")
        res = _result() if i < good else _result(verdict="needs_human_review")
        s = score_case(case, res)
        s.cost_usd = cost
        out.append(s)
    return out


def test_identical_runs_pass_all_gates() -> None:
    a = _record("a", _scores(10, 10))
    rep = compare_runs(a, _record("b", _scores(10, 10)))
    assert rep.passed and all(g.passed for g in rep.gates)
    text = render_comparison(rep)
    assert "Composite score" in text and "PASS" in text and "**Result: PASS**" in text


def test_composite_drop_fails() -> None:
    rep = compare_runs(_record("a", _scores(10, 10)), _record("b", _scores(10, 7)))
    assert not rep.passed
    assert next(g for g in rep.gates if g.name == "composite_drop").passed is False


def test_cost_gate_thresholds() -> None:
    base = _record("a", _scores(10, 10, cost=0.10))
    assert compare_runs(base, _record("b", _scores(10, 10, cost=0.12))).passed
    rep = compare_runs(base, _record("c", _scores(10, 10, cost=0.13)))
    assert not rep.passed and not next(g for g in rep.gates if g.name == "cost_rise").passed
    zero = _record("z", _scores(10, 10, cost=0.0))
    gate = next(g for g in compare_runs(zero, base).gates if g.name == "cost_rise")
    assert gate.passed and "skipped" in gate.detail


def test_compare_rejects_partial_candidate() -> None:
    rep = compare_runs(_record("a", _scores(10, 10)), _record("b", _scores(9, 9)))
    assert not rep.passed
    assert any(g.name == "same_cases" and not g.passed for g in rep.gates)
    relaxed = compare_runs(
        _record("a", _scores(10, 10)), _record("b", _scores(9, 9)), Gates(require_same_cases=False)
    )
    assert relaxed.passed


def test_render_metrics_lists_every_row() -> None:
    text = render_metrics(_record("a", _scores(3, 2)))
    for label in ("Verdict accuracy", "Grounding rate", "Cost / case (USD)", "attack"):
        assert label in text
