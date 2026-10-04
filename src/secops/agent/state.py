"""Typed graph state."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from secops.schemas.agent import CriticIssue, DraftReport, Evidence, ToolCallRecord, TriageReport
from secops.schemas.alert import Alert


class InvestigationState(TypedDict, total=False):
    investigation_id: str
    alert: Alert
    plan: list[str]
    messages: list[dict[str, Any]]  # investigator history, append-only
    evidence: list[Evidence]
    tool_calls: list[ToolCallRecord]
    draft: DraftReport | None
    critic_issues: list[CriticIssue]  # issues from the latest critic round
    all_issues: list[CriticIssue]  # every issue raised during the investigation
    iteration: int  # critic rejections so far
    tool_budget_remaining: int
    status: str
    started_at: datetime
    errors: list[str]
    report: TriageReport | None
