"""Merge repeats from several run files into one k-repeat run (same golden set, same cases).

    uv run python scripts/merge_eval_runs.py --out agent-v1-k3 \\
        --only-from agent-v1-k3extra agent-v1-k1 agent-v1-k3extra

The first run's results (filtered to the case ids present in `--only-from`) become repeat 0,
the next run's repeats are appended with shifted indices, and metrics are recomputed. Usage
and cost are kept per result; nothing is re-run."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from secops.evaluation.metrics import aggregate
from secops.evaluation.runner import CaseResult, RunRecord, load_run

RUNS = Path(__file__).resolve().parents[1] / "evaluation" / "runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run id of the merged file")
    ap.add_argument("--only-from", help="restrict cases to those present in this run id")
    ap.add_argument("runs", nargs="+", help="run ids in order; repeats are appended")
    args = ap.parse_args()
    records = [load_run(RUNS / f"{r}.json") for r in args.runs]
    cases = (
        {r.case_id for r in load_run(RUNS / f"{args.only_from}.json").results}
        if args.only_from
        else {r.case_id for rec in records for r in rec.results}
    )
    merged: list[CaseResult] = []
    offset = 0
    for rec in records:
        subset = [r for r in rec.results if r.case_id in cases]
        n_repeats = max(r.repeat for r in subset) + 1 if subset else 0
        for r in subset:
            score = r.score.model_copy(update={"repeat": r.repeat + offset})
            merged.append(r.model_copy(update={"repeat": r.repeat + offset, "score": score}))
        offset += n_repeats
    base = records[0]
    out = RunRecord(
        config=base.config.model_copy(
            update={"run_id": args.out, "repeats": offset, "only": sorted(cases)}
        ),
        started_at=min(r.started_at for r in records),
        finished_at=datetime.now(UTC),
        prompt_version=base.prompt_version,
        models=base.models,
        golden_version=base.golden_version,
        results=sorted(merged, key=lambda r: (r.case_id, r.repeat)),
        metrics=aggregate([r.score for r in merged]),
    )
    path = RUNS / f"{args.out}.json"
    path.write_text(out.model_dump_json(indent=2))
    m = out.metrics
    assert m is not None
    print(
        f"{args.out}: {m.cases} cases x {m.repeats} repeats, verdict {m.verdict_accuracy.mean:.3f} "
        f"± {m.verdict_accuracy.std:.3f}, composite {m.composite:.3f}, flaky {m.flaky_cases}, "
        f"cost ${m.cost_total_usd:.3f} -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
