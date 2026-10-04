"""Agent endpoints: start an investigation in the background, read it back, list evaluation
runs. One in-process manager per app; a second request for an alert that is still running
returns the existing investigation instead of paying twice."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy import Engine

from secops.agent.cli import make_llms
from secops.agent.graph import AgentDeps, run_investigation
from secops.agent.replay import replay_registry
from secops.agent.repository import InvestigationRepository
from secops.agent.settings import get_agent_settings
from secops.api.auth import require_api_key
from secops.api.detector import DetectorService
from secops.api.ratelimit import rate_limited
from secops.api.settings import ApiSettings
from secops.config import get_settings
from secops.db.session import make_engine, upgrade_to_head
from secops.evaluation.runner import load_run
from secops.observability import get_tracer
from secops.schemas.alert import Alert
from secops.schemas.api import (
    EvaluationRunSummary,
    InvestigationDetail,
    InvestigationRequest,
    InvestigationStatus,
)
from secops.tools.detector import DetectorTool
from secops.tools.registry import ToolRegistry, build_registry

log = logging.getLogger("secops.api")
REPLAY_ATTACK_INDEX = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "attack" / "index_subset.json"
)
router = APIRouter(dependencies=[Depends(require_api_key), Depends(rate_limited)])


class InvestigationManager:
    def __init__(
        self,
        settings: ApiSettings,
        registry: ToolRegistry | None = None,
        database_url: str | None = None,
        service_provider: Callable[[], DetectorService | None] | None = None,
    ) -> None:
        self.settings = settings
        self._registry = registry
        self._service_provider = service_provider
        self._database_url = database_url or get_settings().resolved_database_url()
        self._repo: InvestigationRepository | None = None
        self._lock = threading.Lock()
        self.running: dict[str, str] = {}  # alert_id -> investigation_id
        self.statuses: dict[str, InvestigationStatus] = {}

    # ---- lazy wiring ------------------------------------------------------------------------
    def registry(self) -> ToolRegistry:
        if self._registry is None:
            engine: Engine = make_engine(self._database_url, read_only=True)
            if self.settings.agent_mode == "replay":
                # tools are served from recorded outputs; only the input/output models matter
                from secops.tools.nvd import NvdClient

                self._registry = build_registry(
                    engine=engine,
                    attack_index_path=REPLAY_ATTACK_INDEX,
                    nvd_client=NvdClient(cache_dir=Path(".nvd-replay")),
                    include_detector=False,
                )
            else:
                self._registry = build_registry(engine=engine, include_detector=False)
            service = self._service_provider() if self._service_provider else None
            if service is not None:  # the detector as a tool, same bundles the API serves
                for spec in DetectorTool(service, engine).specs():
                    self._registry.register(spec)
        return self._registry

    def repository(self) -> InvestigationRepository:
        if self._repo is None:
            upgrade_to_head(self._database_url)
            self._repo = InvestigationRepository(make_engine(self._database_url))
        return self._repo

    # ---- lifecycle --------------------------------------------------------------------------
    def submit(self, alert: Alert) -> tuple[InvestigationStatus, bool]:
        """Returns (status, started). started is False when the alert is already running."""
        with self._lock:
            existing = self.running.get(alert.alert_id)
            if existing is not None:
                return self.statuses[existing], False
            inv_id = uuid.uuid4().hex
            st = InvestigationStatus(
                investigation_id=inv_id,
                alert_id=alert.alert_id,
                status="queued",
                created_at=datetime.now(UTC),
            )
            self.running[alert.alert_id] = inv_id
            self.statuses[inv_id] = st
            return st, True

    def run(self, inv_id: str, alert: Alert) -> None:
        st = self.statuses[inv_id]
        st.status = "running"
        try:
            agent_settings = get_agent_settings()
            registry = self.registry()
            fixture_root = self.settings.agent_fixture_root or Path("tests/fixtures/llm")
            scenario = self.settings.agent_scenario
            if self.settings.agent_mode == "replay":
                assert scenario is not None
                registry = replay_registry(registry, fixture_root / scenario / "tools")
            investigator, critic = make_llms(
                self.settings.agent_mode,
                scenario,
                fixture_root,
                agent_settings.effort,
                agent_settings,
            )
            deps = AgentDeps(
                registry=registry,
                investigator=investigator,
                critic=critic,
                tool_budget=agent_settings.tool_budget,
                tracer=get_tracer(),  # one tracer per investigation (Langfuse when configured)
            )
            result = run_investigation(alert, deps, investigation_id=inv_id)
            self.repository().save(result)
            st.status = "done"
            st.verdict = result.report.verdict
            st.severity = result.report.severity
            st.cost_usd = result.usage.cost_usd
        except Exception as e:  # the row must never stay "running"
            log.exception("investigation %s failed", inv_id)
            st.status = "failed"
            st.error = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            with self._lock:
                self.running.pop(alert.alert_id, None)

    def get(self, inv_id: str) -> InvestigationDetail | None:
        st = self.statuses.get(inv_id)
        stored = self.repository().get(inv_id)
        if stored is None and st is None:
            return None
        if stored is not None:
            return InvestigationDetail(
                investigation_id=inv_id,
                alert_id=stored.alert_id,
                status="done",
                verdict=stored.report.verdict,
                severity=stored.report.severity,
                cost_usd=stored.cost_usd,
                created_at=stored.created_at,
                report=stored.report,
            )
        assert st is not None
        return InvestigationDetail(**st.model_dump())

    def list(self, limit: int) -> list[InvestigationStatus]:
        live = [s for s in self.statuses.values() if s.status != "done"]
        stored = [
            InvestigationStatus(
                investigation_id=r.investigation_id,
                alert_id=r.alert_id,
                status="done",
                verdict=r.verdict,
                severity=r.severity,
                cost_usd=r.cost_usd,
                created_at=r.created_at,
            )
            for r in self.repository().recent(limit)
        ]
        return (live + stored)[:limit]


def manager_from_app(request: Request) -> InvestigationManager:
    settings: ApiSettings = request.app.state.settings
    if not settings.investigations_enabled:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "investigations are disabled")
    m: InvestigationManager = request.app.state.investigations
    return m


Manager = Annotated[InvestigationManager, Depends(manager_from_app)]


@router.post("/investigations", response_model=InvestigationStatus, status_code=202)
def start_investigation(
    body: InvestigationRequest, background: BackgroundTasks, manager: Manager
) -> InvestigationStatus:
    st, started = manager.submit(body.alert)
    if started:
        background.add_task(manager.run, st.investigation_id, body.alert)
    return st


@router.get("/investigations/{investigation_id}", response_model=InvestigationDetail)
def get_investigation(investigation_id: str, manager: Manager) -> InvestigationDetail:
    detail = manager.get(investigation_id)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such investigation")
    return detail


@router.get("/investigations", response_model=list[InvestigationStatus])
def list_investigations(manager: Manager, limit: int = 20) -> list[InvestigationStatus]:
    return manager.list(max(1, min(limit, 200)))


@router.get("/evaluation/runs", response_model=list[EvaluationRunSummary])
def evaluation_runs(request: Request) -> list[EvaluationRunSummary]:
    settings: ApiSettings = request.app.state.settings
    out: list[EvaluationRunSummary] = []
    if not settings.evaluation_runs_dir.exists():
        return out
    for path in sorted(settings.evaluation_runs_dir.glob("*.json")):
        try:
            run = load_run(path)
        except Exception:  # a half-written or foreign file must not break the listing
            log.warning("skipping unreadable run file %s", path.name)
            continue
        m = run.metrics
        if m is None:
            continue
        out.append(
            EvaluationRunSummary(
                run_id=run.config.run_id,
                investigator=run.config.investigator,
                finished_at=run.finished_at,
                cases=m.cases,
                repeats=m.repeats,
                composite=m.composite,
                verdict_accuracy=m.verdict_accuracy.mean,
                grounding_rate=m.grounding_rate.mean,
                cost_total_usd=m.cost_total_usd,
            )
        )
    return out
