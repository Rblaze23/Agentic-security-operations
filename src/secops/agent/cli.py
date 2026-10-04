"""`secops-agent`: investigate one alert (scored from the event store or loaded from JSON),
persist the result, and show stored investigations."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import anthropic
import typer
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from secops.agent.graph import AgentDeps, InvestigationResult, run_investigation
from secops.agent.llm import LLM, Effort, Mode
from secops.agent.repository import InvestigationRepository
from secops.agent.settings import AgentSettings, get_agent_settings
from secops.api.detector import DetectorService
from secops.api.settings import get_api_settings
from secops.config import Settings, get_settings
from secops.db.models import Event
from secops.db.session import make_engine, upgrade_to_head
from secops.schemas.alert import Alert
from secops.tools.detector import DetectorTool, request_for_event
from secops.tools.registry import ToolRegistry, build_registry

app = typer.Typer(help="Agentic investigation of detector alerts.", no_args_is_help=True)
log = logging.getLogger("secops.agent")


def _err(msg: str) -> None:
    typer.echo(msg, err=True)


def _repository(url: str) -> InvestigationRepository:
    """Writable engine on the store with migrations applied (idempotent, adds tables only)."""
    upgrade_to_head(url)
    return InvestigationRepository(make_engine(url))


def _build(
    settings: Settings, engine: Engine, with_detector: bool
) -> tuple[ToolRegistry, DetectorService | None]:
    """Registry over the read-only event store; the detector tool only when bundles are wanted."""
    registry = build_registry(settings=settings, engine=engine, include_detector=False)
    service: DetectorService | None = None
    if with_detector:
        api = get_api_settings()
        family = api.family_dir if api.family_dir.exists() else None
        service = DetectorService.from_dirs(api.detector_dir, family, k=api.top_k)
        for spec in DetectorTool(service, engine).specs():
            registry.register(spec)
    return registry, service


def alert_from_event(event_id: int, engine: Engine, service: DetectorService) -> Alert:
    """Score one stored flow with the detector and wrap the prediction as an Alert."""
    with Session(engine) as s:
        event = s.get(Event, event_id)
        if event is None:
            raise KeyError(f"event {event_id} is not in the event store")
        request = request_for_event(event, service.spec)
    prediction = service.predict([request])[0]
    return Alert.from_prediction(request, prediction)


def make_llms(
    mode: Mode,
    scenario: str | None,
    fixture_root: Path,
    effort: Effort,
    settings: AgentSettings,
) -> tuple[LLM, LLM]:
    """Investigator and critic adapters. Replay needs no key; live and record need the key."""
    client: Any | None = None
    if mode != "replay":
        key = settings.anthropic_api_key
        if key is None or not key.get_secret_value():
            raise typer.BadParameter("ANTHROPIC_API_KEY is not set (environment or .env)")
        client = anthropic.Anthropic(api_key=key.get_secret_value())
    fixture_dir: Path | None = None
    if mode != "live":
        if not scenario:
            raise typer.BadParameter("--scenario is required for record and replay modes")
        fixture_dir = fixture_root / scenario
    investigator = LLM(
        settings.investigator_model,
        effort=effort,
        mode=mode,
        fixture_dir=fixture_dir / "investigator" if fixture_dir else None,
        client=client,
    )
    critic = LLM(
        settings.critic_model,
        effort=effort,
        mode=mode,
        fixture_dir=fixture_dir / "critic" if fixture_dir else None,
        client=client,
    )
    return investigator, critic


def _summary_lines(result: InvestigationResult) -> list[str]:
    r = result.report
    u = result.usage
    lines = [
        f"investigation_id: {result.investigation_id}",
        f"verdict: {r.verdict}  severity: {r.severity}  confidence: {r.confidence:.2f}  "
        f"family: {r.attack_family or '-'}",
        f"summary: {r.summary}",
        f"findings: {len(r.findings)}  techniques: {len(r.attack_techniques)}  "
        f"cves: {len(r.cves)}  actions: {len(r.recommended_actions)}  "
        f"uncertainties: {len(r.uncertainties)}",
        f"tool calls: {len(result.tool_calls)} (budget left {result.tool_budget_remaining})  "
        f"critic rejections: {result.iteration}  status: {result.status}",
        f"usage: {u.calls} calls, {u.input_tokens} in / {u.output_tokens} out tokens, "
        f"cost ${u.cost_usd:.4f}, {result.latency_ms / 1000:.1f} s",
    ]
    for model, m in u.by_model.items():
        lines.append(
            f"  {model}: {m.calls} calls, {m.input_tokens} in, {m.output_tokens} out, "
            f"{m.cache_read_tokens} cache read, ${m.cost_usd:.4f}"
        )
    return lines


@app.command()
def investigate(
    event_id: Annotated[
        int | None, typer.Option(help="Score this stored flow and investigate it.")
    ] = None,
    alert_json: Annotated[
        Path | None, typer.Option(help="Investigate an Alert JSON file instead.")
    ] = None,
    mode: Annotated[str, typer.Option(help="live | record | replay")] = "live",
    scenario: Annotated[
        str | None, typer.Option(help="Fixture scenario name for record/replay.")
    ] = None,
    fixture_root: Annotated[Path, typer.Option(help="Fixture root directory.")] = Path(
        "tests/fixtures/llm"
    ),
    budget: Annotated[
        int | None, typer.Option(help="Tool-call budget (default from settings).")
    ] = None,
    effort: Annotated[str | None, typer.Option(help="low | medium | high | max")] = None,
    persist: Annotated[
        bool, typer.Option("--persist/--no-persist", help="Store the result.")
    ] = True,
    detector_tool: Annotated[
        bool,
        typer.Option(
            "--detector-tool/--no-detector-tool", help="Register predict_attack (needs bundles)."
        ),
    ] = True,
    out: Annotated[
        Path | None, typer.Option(help="Write the report JSON here instead of stdout.")
    ] = None,
    database_url: Annotated[str | None, typer.Option(help="Override SECOPS_DATABASE_URL.")] = None,
) -> None:
    """Run one investigation and print the triage report as JSON (summary on stderr)."""
    if (event_id is None) == (alert_json is None):
        raise typer.BadParameter("provide exactly one of --event-id or --alert-json")
    if mode not in ("live", "record", "replay"):
        raise typer.BadParameter("--mode must be live, record or replay")
    settings = get_settings()
    agent_settings = get_agent_settings()
    url = database_url or settings.resolved_database_url()
    engine_ro = make_engine(url, read_only=True)
    with_detector = detector_tool or event_id is not None
    registry, service = _build(settings, engine_ro, with_detector)

    if alert_json is not None:
        alert = Alert.model_validate_json(alert_json.read_text())
    else:
        assert event_id is not None and service is not None
        try:
            alert = alert_from_event(event_id, engine_ro, service)
        except KeyError as e:
            _err(str(e))
            raise typer.Exit(code=1) from e
        if not alert.prediction.is_alert:
            _err(
                f"note: event {event_id} scores {alert.prediction.attack_probability:.4f}, below "
                f"the threshold {alert.prediction.threshold:.6f}; investigating anyway"
            )

    investigator, critic = make_llms(
        mode,  # type: ignore[arg-type]
        scenario,
        fixture_root,
        effort or agent_settings.effort,  # type: ignore[arg-type]
        agent_settings,
    )
    deps = AgentDeps(
        registry=registry,
        investigator=investigator,
        critic=critic,
        tool_budget=budget or agent_settings.tool_budget,
    )
    result = run_investigation(alert, deps, investigation_id=scenario if mode != "live" else None)

    report_json = result.report.model_dump_json(indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_json)
        _err(f"report written to {out}")
    else:
        typer.echo(report_json)
    for line in _summary_lines(result):
        _err(line)
    if persist:
        _repository(url).save(result)
        _err("persisted.")


@app.command()
def show(
    investigation_id: str,
    database_url: Annotated[str | None, typer.Option(help="Override SECOPS_DATABASE_URL.")] = None,
) -> None:
    """Print a stored investigation: report JSON, tool calls and cost."""
    url = database_url or get_settings().resolved_database_url()
    stored = _repository(url).get(investigation_id)
    if stored is None:
        _err(f"no investigation {investigation_id}")
        raise typer.Exit(code=1)
    typer.echo(stored.report.model_dump_json(indent=2))
    _err(f"investigation_id: {stored.investigation_id}  created: {stored.created_at.isoformat()}")
    _err(
        f"models: {stored.model_investigator} / {stored.model_critic}  "
        f"prompt {stored.prompt_version}"
    )
    _err(f"tool calls ({len(stored.tool_calls)}):")
    for c in stored.tool_calls:
        args = json.dumps(c.arguments, default=str)[:80]
        _err(f"  {c.evidence_id:>4}  {c.tool:<24} {c.status:<16} {c.latency_ms:8.1f} ms  {args}")
    _err(
        f"usage: {stored.input_tokens} in / {stored.output_tokens} out / "
        f"{stored.cache_read_tokens} cache read tokens, cost ${stored.cost_usd:.4f}, "
        f"{stored.latency_ms / 1000:.1f} s, {stored.iterations} critic rejections"
    )


@app.command()
def recent(
    limit: Annotated[int, typer.Option(min=1, max=200)] = 20,
    database_url: Annotated[str | None, typer.Option(help="Override SECOPS_DATABASE_URL.")] = None,
) -> None:
    """List the most recent stored investigations."""
    url = database_url or get_settings().resolved_database_url()
    rows = _repository(url).recent(limit)
    if not rows:
        typer.echo("no investigations stored")
        return
    typer.echo(
        f"{'created':<20} {'id':<32} {'event':>8} {'verdict':<19} {'sev':<8} "
        f"{'family':<13} {'calls':>5} {'cost':>8}"
    )
    for r in rows:
        typer.echo(
            f"{r.created_at.strftime('%Y-%m-%d %H:%M:%S'):<20} {r.investigation_id:<32} "
            f"{r.event_id if r.event_id is not None else '-':>8} {r.verdict:<19} {r.severity:<8} "
            f"{r.attack_family or '-':<13} {r.tool_call_count:>5} ${r.cost_usd:>7.4f}"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
    )
    app()


if __name__ == "__main__":
    main()
