"""`secops-eval`: golden set, runs, comparison, reports."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from secops.agent.cli import _build
from secops.config import get_settings
from secops.db.session import make_engine
from secops.evaluation.golden import build_golden_set

app = typer.Typer(help="Evaluation of the investigation agent.", no_args_is_help=True)
GOLDEN_DIR = Path("evaluation/golden")


@app.callback()
def _root() -> None:
    """Golden set, runs, comparison and reports for the investigation agent."""


@app.command("build-golden")
def build_golden(
    out: Annotated[Path, typer.Option(help="Where to write the golden set JSON.")] = GOLDEN_DIR
    / "v1.json",
    seed: Annotated[int, typer.Option()] = 42,
    per_family: Annotated[int, typer.Option(min=1, max=10)] = 4,
    benign: Annotated[int, typer.Option(min=1, max=20)] = 6,
    adversarial: Annotated[int, typer.Option(min=0, max=10)] = 4,
) -> None:
    """Sample test-split alerts per family plus benign false positives; derive expectations."""
    settings = get_settings()
    engine = make_engine(settings.resolved_database_url(), read_only=True)
    _registry, service = _build(settings, engine, with_detector=True)
    assert service is not None
    gs = build_golden_set(
        engine, service, seed=seed, per_family=per_family, benign=benign, adversarial=adversarial
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(gs.model_dump_json(indent=2))
    by_kind = {
        k: sum(1 for c in gs.cases if c.kind == k) for k in ("attack", "benign_fp", "adversarial")
    }
    typer.echo(f"wrote {len(gs.cases)} cases to {out}: {by_kind}")


@app.command()
def run(
    run_id: Annotated[str, typer.Option(help="Name of the run file under evaluation/runs/.")],
    golden: Annotated[Path, typer.Option()] = GOLDEN_DIR / "v1.json",
    investigator: Annotated[str, typer.Option(help="agent | baseline")] = "agent",
    mode: Annotated[str, typer.Option(help="live | record | replay")] = "record",
    recordings: Annotated[
        Path | None,
        typer.Option(help="Fixture root (default $SECOPS_DATA_DIR/eval/recordings/<run_id>)."),
    ] = None,
    repeats: Annotated[int, typer.Option(min=1, max=5)] = 1,
    only: Annotated[list[str] | None, typer.Option(help="Case ids to run.")] = None,
    tool_budget: Annotated[int, typer.Option(min=1, max=50)] = 12,
    judge: Annotated[
        bool, typer.Option("--judge/--no-judge", help="Sonnet judge per observed finding.")
    ] = False,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
    no_detector_tool: Annotated[
        bool, typer.Option("--no-detector-tool", help="Skip predict_attack (no bundles).")
    ] = False,
    llm_critic: Annotated[
        bool, typer.Option("--llm-critic/--no-llm-critic", help="Ablation: rules-only critic.")
    ] = True,
    out_dir: Annotated[Path, typer.Option()] = Path("evaluation/runs"),
) -> None:
    """Run the agent or the baseline over the golden set; writes evaluation/runs/<run_id>.json."""
    from secops.evaluation.compare import render_metrics
    from secops.evaluation.runner import RunConfig, run_golden

    if investigator not in ("agent", "baseline"):
        raise typer.BadParameter("--investigator must be agent or baseline")
    if mode not in ("live", "record", "replay"):
        raise typer.BadParameter("--mode must be live, record or replay")
    settings = get_settings()
    rec_dir = recordings or settings.data_dir / "eval" / "recordings" / run_id
    config = RunConfig(
        run_id=run_id,
        golden_path=golden,
        investigator=investigator,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        recordings_dir=rec_dir,
        repeats=repeats,
        tool_budget=tool_budget,
        judge=judge,
        only=only or None,
        llm_critic=llm_critic,
    )
    engine = make_engine(settings.resolved_database_url(), read_only=True)
    registry, service = _build(settings, engine, with_detector=not no_detector_tool)
    from secops.evaluation.golden import load_golden_set
    from secops.evaluation.runner import estimated_cost_usd

    n = len([c for c in load_golden_set(golden).cases if not only or c.case_id in only])
    typer.echo(
        f"run {run_id}: {n} cases x {repeats} repeats, {investigator} in {mode} mode; "
        f"estimated API cost ${estimated_cost_usd(config, n):.2f}",
        err=True,
    )
    out = out_dir / f"{run_id}.json"
    record = run_golden(config, engine, registry, service, out, resume=resume)
    typer.echo(render_metrics(record))
    assert record.metrics is not None
    typer.echo(f"measured cost ${record.metrics.cost_total_usd:.4f}; written to {out}", err=True)


@app.command()
def compare(
    baseline: Path,
    candidate: Path,
    composite_max_drop: Annotated[float, typer.Option()] = 0.02,
    grounding_max_drop: Annotated[float, typer.Option()] = 0.02,
    cost_max_rise: Annotated[float, typer.Option()] = 0.25,
    same_cases: Annotated[bool, typer.Option("--same-cases/--allow-partial")] = True,
) -> None:
    """Before/after table for two run files; exit 1 on any failed gate."""
    from secops.evaluation.compare import Gates, compare_runs, render_comparison
    from secops.evaluation.runner import load_run

    report = compare_runs(
        load_run(baseline),
        load_run(candidate),
        Gates(
            composite_max_drop=composite_max_drop,
            grounding_max_drop=grounding_max_drop,
            cost_max_rise=cost_max_rise,
            require_same_cases=same_cases,
        ),
    )
    typer.echo(render_comparison(report))
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def report(run: Path) -> None:
    """Print a run's metrics table (markdown)."""
    from secops.evaluation.compare import render_metrics
    from secops.evaluation.runner import load_run

    typer.echo(render_metrics(load_run(run)))


@app.command()
def rescore(
    run: Path,
    no_detector_tool: Annotated[
        bool, typer.Option("--no-detector-tool", help="Skip predict_attack (no bundles).")
    ] = False,
) -> None:
    """Recompute a recorded run's scores from its recordings (zero cost); keeps measured latency."""
    from secops.evaluation.compare import render_metrics
    from secops.evaluation.runner import rescore_run

    settings = get_settings()
    engine = make_engine(settings.resolved_database_url(), read_only=True)
    registry, _service = _build(settings, engine, with_detector=not no_detector_tool)
    record = rescore_run(run, engine, registry)
    typer.echo(render_metrics(record))
    typer.echo(f"rescored {run} at {record.rescored_at}", err=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
    )
    app()


if __name__ == "__main__":
    main()
