"""Run the agent or the rule-based baseline over the golden set, k repeats, per-case
record/replay, incremental output and resume."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine

from secops.agent.cli import alert_from_event, make_llms
from secops.agent.graph import AgentDeps, InvestigationResult, run_investigation
from secops.agent.llm import LLM, Effort
from secops.agent.prompts import PROMPT_VERSION
from secops.agent.replay import recording_registry, replay_registry
from secops.agent.settings import AgentSettings, get_agent_settings
from secops.api.detector import DetectorService
from secops.evaluation.adversarial import adversarial_registry
from secops.evaluation.baseline import PROMPT_VERSION as BASELINE_VERSION
from secops.evaluation.baseline import RuleBasedInvestigator
from secops.evaluation.golden import GoldenCase, load_golden_set
from secops.evaluation.metrics import CaseScore, RunMetrics, aggregate, score_case
from secops.schemas.agent import TriageReport
from secops.schemas.alert import Alert
from secops.tools.registry import ToolRegistry

log = logging.getLogger(__name__)
COST_PER_INVESTIGATION_USD = 0.20  # Phase 4 measured mean, docs/agent.md
COST_PER_JUDGE_CALL_USD = 0.02


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    golden_path: Path
    investigator: Literal["agent", "baseline"] = "agent"
    mode: Literal["live", "record", "replay"] = "record"
    recordings_dir: Path
    repeats: int = 1
    tool_budget: int = 12
    effort: Effort = "medium"
    judge: bool = False
    only: list[str] | None = None
    compress: bool = True
    llm_critic: bool = True


class CaseResult(BaseModel):
    case_id: str
    repeat: int
    score: CaseScore
    report: TriageReport
    errors: list[str]


class RunRecord(BaseModel):
    config: RunConfig
    started_at: datetime
    finished_at: datetime | None
    prompt_version: str
    models: dict[str, str]
    golden_version: str
    results: list[CaseResult]
    metrics: RunMetrics | None
    rescored_at: datetime | None = None  # set by rescore_run: scores recomputed from recordings


def estimated_cost_usd(config: RunConfig, n_cases: int) -> float:
    if config.investigator == "baseline" or config.mode == "replay":
        return 0.0
    per_case = COST_PER_INVESTIGATION_USD + (COST_PER_JUDGE_CALL_USD if config.judge else 0.0)
    return n_cases * config.repeats * per_case


def _case_dir(config: RunConfig, case: GoldenCase, repeat: int) -> Path:
    return config.recordings_dir / case.case_id / f"r{repeat}"


def _alert_for(
    case: GoldenCase,
    case_dir: Path,
    config: RunConfig,
    engine: Engine,
    service: DetectorService | None,
) -> Alert:
    saved = case_dir / "alert.json"
    if config.mode == "replay" and saved.exists():
        return Alert.model_validate_json(saved.read_text())
    if service is None:
        raise RuntimeError("a DetectorService is required to build alerts outside replay mode")
    try:
        alert = alert_from_event(case.event_id, engine, service)
    except KeyError as e:
        raise KeyError(f"{case.case_id}: {e}") from e
    if config.mode == "record":
        case_dir.mkdir(parents=True, exist_ok=True)
        saved.write_text(alert.model_dump_json(indent=2))
    return alert


def _registry_for(
    config: RunConfig, case: GoldenCase, case_dir: Path, base: ToolRegistry
) -> ToolRegistry:
    reg = adversarial_registry(base, case.injection) if case.injection else base
    if config.mode == "record":
        return recording_registry(reg, case_dir / "tools", compress=config.compress)
    if config.mode == "replay":
        return replay_registry(reg, case_dir / "tools")
    return reg


def _judge(config: RunConfig, case_dir: Path, settings: AgentSettings) -> LLM | None:
    if not config.judge or config.investigator != "agent":
        return None
    _investigator, critic = make_llms(
        config.mode, "judge", case_dir, settings.effort, settings, compress=config.compress
    )
    return critic


def run_case(
    config: RunConfig,
    case: GoldenCase,
    repeat: int,
    engine: Engine,
    base_registry: ToolRegistry,
    service: DetectorService | None,
    settings: AgentSettings,
) -> tuple[InvestigationResult, CaseScore]:
    case_dir = _case_dir(config, case, repeat)
    alert = _alert_for(case, case_dir, config, engine, service)
    registry = _registry_for(config, case, case_dir, base_registry)
    investigation_id = f"{case.case_id}-r{repeat}"
    if config.investigator == "baseline":
        result = RuleBasedInvestigator(registry, budget=config.tool_budget).investigate(
            alert, investigation_id=investigation_id
        )
    else:
        investigator, critic = make_llms(
            config.mode,
            "run",
            case_dir,
            config.effort,
            settings,
            compress=config.compress,
        )
        deps = AgentDeps(
            registry=registry,
            investigator=investigator,
            critic=critic,
            tool_budget=config.tool_budget,
            llm_critic=config.llm_critic,
        )
        result = run_investigation(alert, deps, investigation_id=investigation_id)
    score = score_case(case, result, judge=_judge(config, case_dir / "run", settings))
    score.repeat = repeat
    return result, score


def run_golden(
    config: RunConfig,
    engine: Engine,
    registry: ToolRegistry,
    service: DetectorService | None,
    out_path: Path,
    resume: bool = False,
    settings: AgentSettings | None = None,
) -> RunRecord:
    settings = settings or get_agent_settings()
    golden = load_golden_set(config.golden_path)
    cases = [c for c in golden.cases if not config.only or c.case_id in config.only]
    if config.only:
        missing = sorted(set(config.only) - {c.case_id for c in cases})
        if missing:
            raise KeyError(f"unknown golden case ids: {missing}")
    if resume and out_path.exists():
        record = RunRecord.model_validate_json(out_path.read_text())
        record.finished_at = None
    else:
        models = (
            {"investigator": "rule-based", "critic": "none"}
            if config.investigator == "baseline"
            else {"investigator": settings.investigator_model, "critic": settings.critic_model}
        )
        record = RunRecord(
            config=config,
            started_at=datetime.now(UTC),
            finished_at=None,
            prompt_version=BASELINE_VERSION
            if config.investigator == "baseline"
            else PROMPT_VERSION,
            models=models,
            golden_version=golden.version,
            results=[],
            metrics=None,
        )
    done = {(r.case_id, r.repeat) for r in record.results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for repeat in range(config.repeats):
            if (case.case_id, repeat) in done:
                continue
            result, score = run_case(config, case, repeat, engine, registry, service, settings)
            record.results.append(
                CaseResult(
                    case_id=case.case_id,
                    repeat=repeat,
                    score=score,
                    report=result.report,
                    errors=list(result.errors),
                )
            )
            out_path.write_text(record.model_dump_json(indent=2))
            log.info(
                "%s r%d: %s/%s cost $%.4f",
                case.case_id,
                repeat,
                score.verdict,
                score.severity,
                score.cost_usd,
            )
    record.metrics = aggregate([r.score for r in record.results])
    record.finished_at = datetime.now(UTC)
    out_path.write_text(record.model_dump_json(indent=2))
    return record


def load_run(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(path.read_text())


def run_summary(record: RunRecord) -> dict[str, Any]:
    m = record.metrics
    assert m is not None
    return {
        "run_id": record.config.run_id,
        "cases": m.cases,
        "repeats": m.repeats,
        "composite": m.composite,
        "verdict_accuracy": m.verdict_accuracy.mean,
        "grounding_rate": m.grounding_rate.mean,
        "cost_total_usd": m.cost_total_usd,
    }


def rescore_run(
    path: Path,
    engine: Engine,
    registry: ToolRegistry,
    settings: AgentSettings | None = None,
) -> RunRecord:
    """Recompute every score of a recorded run from its recordings (zero cost) after a scorer
    change. Measured latency and cost are what the original run observed, so they are kept;
    verdicts, reports and all other score fields come from the replay."""
    settings = settings or get_agent_settings()
    record = load_run(path)
    replay_cfg = record.config.model_copy(update={"mode": "replay"})
    golden = {c.case_id: c for c in load_golden_set(record.config.golden_path).cases}
    new_results: list[CaseResult] = []
    for old in record.results:
        case = golden[old.case_id]
        result, score = run_case(replay_cfg, case, old.repeat, engine, registry, None, settings)
        score.latency_ms = old.score.latency_ms
        new_results.append(
            CaseResult(
                case_id=old.case_id,
                repeat=old.repeat,
                score=score,
                report=result.report,
                errors=list(result.errors),
            )
        )
    record.results = new_results
    record.metrics = aggregate([r.score for r in new_results])
    record.rescored_at = datetime.now(UTC)
    path.write_text(record.model_dump_json(indent=2))
    return record
