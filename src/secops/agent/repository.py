"""Persistence of investigations (migration 0002). Written after the graph finishes, read by
the CLI now and by the Phase 6 API later."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Engine, select

from secops.agent.graph import InvestigationResult
from secops.db.models import Investigation, ToolCall
from secops.db.session import session_scope
from secops.schemas.agent import ToolCallRecord, TriageReport


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@dataclass(frozen=True)
class StoredInvestigation:
    investigation_id: str
    alert_id: str
    event_id: int | None
    created_at: datetime
    status: str
    iterations: int
    report: TriageReport
    tool_calls: list[ToolCallRecord]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    latency_ms: float
    model_investigator: str
    model_critic: str
    prompt_version: str


@dataclass(frozen=True)
class InvestigationSummary:
    investigation_id: str
    created_at: datetime
    alert_id: str
    event_id: int | None
    verdict: str
    severity: str
    attack_family: str | None
    cost_usd: float
    tool_call_count: int


def _event_int(event_id: str | None) -> int | None:
    return int(event_id) if event_id is not None and event_id.isdigit() else None


class InvestigationRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def save(self, result: InvestigationResult, now: datetime | None = None) -> str:
        r = result.report
        u = result.usage
        by_evidence = {e.evidence_id: e for e in result.evidence}
        row = Investigation(
            investigation_id=result.investigation_id,
            alert_id=result.alert_id or r.alert_id,
            event_id=_event_int(result.event_id),
            status=result.status,
            verdict=r.verdict,
            severity=r.severity,
            confidence=r.confidence,
            attack_family=r.attack_family,
            iterations=result.iteration,
            tool_call_count=len(result.tool_calls),
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=sum(m.cache_read_tokens for m in u.by_model.values()),
            cost_usd=u.cost_usd,
            latency_ms=result.latency_ms,
            model_investigator=result.model_investigator,
            model_critic=result.model_critic,
            prompt_version=result.prompt_version,
            report_json=r.model_dump_json(),
            created_at=now or datetime.now(UTC),
        )
        for c in result.tool_calls:
            ev = by_evidence.get(c.evidence_id)
            row.tool_calls.append(
                ToolCall(
                    evidence_id=c.evidence_id,
                    tool=c.tool,
                    arguments_json=json.dumps(c.arguments, sort_keys=True, default=str),
                    status=c.status,
                    latency_ms=c.latency_ms,
                    called_at=ev.retrieved_at if ev else None,
                )
            )
        with session_scope(self.engine) as s:
            s.add(row)
        return result.investigation_id

    def get(self, investigation_id: str) -> StoredInvestigation | None:
        with session_scope(self.engine) as s:
            row = s.get(Investigation, investigation_id)
            if row is None:
                return None
            calls = [
                ToolCallRecord(
                    tool=c.tool,
                    arguments=json.loads(c.arguments_json),
                    evidence_id=c.evidence_id,
                    status=c.status,  # type: ignore[arg-type]
                    latency_ms=c.latency_ms,
                )
                for c in row.tool_calls
            ]
            created = _aware(row.created_at)
            assert created is not None
            return StoredInvestigation(
                investigation_id=row.investigation_id,
                alert_id=row.alert_id,
                event_id=row.event_id,
                created_at=created,
                status=row.status,
                iterations=row.iterations,
                report=TriageReport.model_validate_json(row.report_json),
                tool_calls=calls,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cache_read_tokens=row.cache_read_tokens,
                cost_usd=row.cost_usd,
                latency_ms=row.latency_ms,
                model_investigator=row.model_investigator,
                model_critic=row.model_critic,
                prompt_version=row.prompt_version,
            )

    def recent(self, limit: int = 20) -> list[InvestigationSummary]:
        stmt = select(Investigation).order_by(Investigation.created_at.desc()).limit(limit)
        with session_scope(self.engine) as s:
            rows = s.scalars(stmt).all()
            out = []
            for row in rows:
                created = _aware(row.created_at)
                assert created is not None
                out.append(
                    InvestigationSummary(
                        investigation_id=row.investigation_id,
                        created_at=created,
                        alert_id=row.alert_id,
                        event_id=row.event_id,
                        verdict=row.verdict,
                        severity=row.severity,
                        attack_family=row.attack_family,
                        cost_usd=row.cost_usd,
                        tool_call_count=row.tool_call_count,
                    )
                )
            return out
