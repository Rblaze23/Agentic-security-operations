"""The investigation graph: plan -> investigate -> critic -> finalize, with bounded loops."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import anthropic
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from secops.agent.critic import check_report, llm_check
from secops.agent.llm import LLM, LLMRefusalError, LLMTruncatedError
from secops.agent.prompts import PROMPT_VERSION, load_prompt
from secops.agent.rubric import severity_for
from secops.agent.state import InvestigationState
from secops.agent.tools import ToolExecutor, tool_definitions
from secops.schemas.agent import (
    CriticIssue,
    DraftReport,
    Evidence,
    ModelPredictionSummary,
    Plan,
    ToolCallRecord,
    TriageReport,
    UsageTotals,
)
from secops.schemas.alert import Alert
from secops.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_CRITIC_REJECTIONS = 2
DEFAULT_PLAN = [
    "Did the source produce a burst of flows to many ports or hosts around the alert time?",
    "What is the destination asset and how critical is it?",
    "Is the source a known attacker address or an internal host?",
]


@dataclass
class AgentDeps:
    registry: ToolRegistry
    investigator: LLM
    critic: LLM
    tool_budget: int = 12
    max_investigate_rounds: int | None = None  # defaults to tool_budget + 3


@dataclass
class InvestigationResult:
    investigation_id: str
    report: TriageReport
    plan: list[str]
    evidence: list[Evidence]
    tool_calls: list[ToolCallRecord]
    messages: list[dict[str, Any]]
    critic_issues: list[CriticIssue]
    all_issues: list[CriticIssue]
    iteration: int
    tool_budget_remaining: int
    status: str
    usage: UsageTotals
    latency_ms: float
    prompt_version: str = PROMPT_VERSION
    errors: list[str] = field(default_factory=list)
    alert_id: str = ""
    event_id: str | None = None
    model_investigator: str = ""
    model_critic: str = ""


# ---- helpers ------------------------------------------------------------------------------
def _alert_brief(alert: Alert) -> dict[str, Any]:
    p = alert.prediction
    return {
        "alert_id": alert.alert_id,
        "event_id": alert.event_id,
        "metadata": alert.metadata.model_dump(mode="json"),
        "detector": {
            "attack_probability": p.attack_probability,
            "threshold": p.threshold,
            "predicted_family": p.predicted_family,
            "family_probabilities": p.family_probabilities,
            "top_contributions": [c.model_dump() for c in p.top_contributions],
            "model": f"{p.model_name} v{p.model_version}",
        },
    }


def _tool_catalogue(registry: ToolRegistry) -> list[dict[str, str]]:
    return [
        {"name": s.name, "description": s.description.split(". ")[0] + "."} for s in registry.all()
    ]


def _asset_criticality(alert: Alert, evidence: list[Evidence]) -> str | None:
    dst = str(alert.metadata.destination_ip) if alert.metadata.destination_ip else None
    for ev in evidence:
        if ev.kind != "tool_result":
            continue
        asset = ev.payload.get("asset") if ev.tool in ("get_asset", "enrich_ip") else None
        if asset and (dst is None or asset.get("ip") == dst):
            return str(asset.get("criticality"))
    return None


# ---- nodes --------------------------------------------------------------------------------
def build_graph(deps: AgentDeps) -> Any:
    defs = tool_definitions(deps.registry)
    investigator_system = load_prompt("investigator")
    planner_system = load_prompt("planner")
    executors: dict[str, ToolExecutor] = {}

    def executor_for(state: InvestigationState) -> ToolExecutor:
        inv_id = state["investigation_id"]
        if inv_id not in executors:
            executors[inv_id] = ToolExecutor(deps.registry, budget=deps.tool_budget)
        return executors[inv_id]

    def plan_node(state: InvestigationState) -> dict[str, Any]:
        alert = state["alert"]
        content = json.dumps(
            {"alert": _alert_brief(alert), "tools": _tool_catalogue(deps.registry)}, default=str
        )
        try:
            resp = deps.investigator.create(
                system=planner_system,
                messages=[{"role": "user", "content": content}],
                output_model=Plan,
            )
            questions = Plan.model_validate(resp.parsed).questions
            errors: list[str] = []
        except (LLMRefusalError, LLMTruncatedError, ValidationError, anthropic.APIError) as e:
            log.warning("planner failed (%s: %s); using the default plan", type(e).__name__, e)
            questions = list(DEFAULT_PLAN)
            errors = [f"planner fallback: {type(e).__name__}: {str(e)[:200]}"]
        first_turn = json.dumps(
            {
                "task": "Investigate this alert and answer with the required JSON object "
                "when done.",
                "alert": _alert_brief(alert),
                "plan": questions,
                "tool_budget": deps.tool_budget,
            },
            default=str,
        )
        return {
            "plan": questions,
            "messages": [{"role": "user", "content": first_turn}],
            "status": "investigating",
            "errors": state.get("errors", []) + errors,
        }

    def investigate_node(state: InvestigationState) -> dict[str, Any]:
        ex = executor_for(state)
        messages = list(state["messages"])
        evidence = list(state.get("evidence", []))
        max_rounds = deps.max_investigate_rounds or deps.tool_budget + 3
        draft: DraftReport | None = None
        errors = list(state.get("errors", []))
        for _ in range(max_rounds):
            tools = defs if ex.remaining > 0 else None
            try:
                resp = deps.investigator.create(
                    system=investigator_system,
                    messages=messages,
                    tools=tools,
                    output_model=DraftReport,
                )
            except (LLMRefusalError, LLMTruncatedError, anthropic.APIError) as e:
                log.error("investigator failed (%s: %s)", type(e).__name__, e)
                errors.append(f"investigator failed: {type(e).__name__}: {str(e)[:200]}")
                return {
                    "messages": messages,
                    "evidence": evidence,
                    "draft": None,
                    "tool_calls": list(ex.records),
                    "tool_budget_remaining": ex.remaining,
                    "status": "failed",
                    "errors": errors,
                }
            messages.append(resp.assistant_message)
            if resp.stop_reason == "tool_use" and resp.tool_uses:
                results = []
                for tu in resp.tool_uses:
                    block, ev = ex.execute(tu)
                    evidence.append(ev)
                    results.append(block)
                messages.append({"role": "user", "content": results})
                continue
            try:
                draft = DraftReport.model_validate(resp.parsed or {})
            except ValidationError as e:
                errors.append(f"draft invalid: {e.error_count()} errors")
                messages.append(
                    {
                        "role": "user",
                        "content": f"The JSON was invalid: {e.errors()[:3]}. "
                        "Answer again with a valid object.",
                    }
                )
                continue
            break
        status = "reviewing" if draft is not None else "failed"
        if draft is None and not errors:
            errors.append("investigator did not produce a report within the round limit")
        return {
            "messages": messages,
            "evidence": evidence,
            "draft": draft,
            "tool_calls": list(ex.records),
            "tool_budget_remaining": ex.remaining,
            "status": status,
            "errors": errors,
        }

    def critic_node(state: InvestigationState) -> dict[str, Any]:
        draft = state.get("draft")
        if draft is None:
            return {"critic_issues": [], "status": "failed"}
        by_id = {e.evidence_id: e for e in state.get("evidence", [])}
        issues = check_report(draft, by_id)
        if not issues:
            issues = llm_check(deps.critic, draft, by_id)
        iteration = state.get("iteration", 0)
        update: dict[str, Any] = {
            "critic_issues": issues,
            "all_issues": state.get("all_issues", []) + issues,
        }
        if issues:
            iteration += 1
            feedback = json.dumps(
                {
                    "critic": "The report was rejected. Fix every issue below, citing only "
                    "evidence "
                    "ids you received, then answer again with the JSON object.",
                    "issues": [i.model_dump() for i in issues],
                },
                default=str,
            )
            update["messages"] = state["messages"] + [{"role": "user", "content": feedback}]
        update["iteration"] = iteration
        return update

    def route_after_investigate(state: InvestigationState) -> Literal["critic", "finalize"]:
        return "critic" if state.get("status") == "reviewing" else "finalize"

    def route_after_critic(state: InvestigationState) -> Literal["investigate", "finalize"]:
        if not state.get("critic_issues"):
            return "finalize"
        return "investigate" if state.get("iteration", 0) < MAX_CRITIC_REJECTIONS else "finalize"

    def finalize_node(state: InvestigationState) -> dict[str, Any]:
        alert = state["alert"]
        draft = state.get("draft")
        evidence = state.get("evidence", [])
        issues = state.get("critic_issues", [])
        forced_review = draft is None or bool(issues)
        verdict = "needs_human_review" if forced_review else draft.verdict  # type: ignore[union-attr]
        family = (draft.attack_family if draft else None) or alert.prediction.predicted_family
        severity = severity_for(
            family,
            _asset_criticality(alert, evidence),
            bool(draft and draft.success_indicator),
            verdict,
        )
        uncertainties = list(draft.uncertainties) if draft else []
        uncertainties += [
            f"{e.tool} ({e.evidence_id}) failed: {e.summary}"
            for e in evidence
            if e.kind == "tool_error"
        ]
        uncertainties += [
            f"{e.tool} ({e.evidence_id}) returned {e.payload.get('status')}"
            for e in evidence
            if e.kind == "tool_result" and e.payload.get("status") in ("unavailable", "not_found")
        ]
        uncertainties += [f"critic: {i.message}" for i in issues]
        uncertainties += [f"error: {e}" for e in state.get("errors", [])]
        steps = [
            f"{r.tool}({json.dumps(r.arguments, default=str)}) -> {r.evidence_id} [{r.status}]"
            for r in state.get("tool_calls", [])
        ]
        p = alert.prediction
        report = TriageReport(
            alert_id=alert.alert_id,
            verdict=verdict,
            attack_family=family
            if verdict != "false_positive"
            else (draft.attack_family if draft else None),
            severity=severity,
            confidence=draft.confidence if (draft and not forced_review) else 0.0,
            summary=draft.summary
            if draft
            else "Investigation did not produce a report; human review required.",
            findings=list(draft.findings) if draft else [],
            attack_techniques=list(draft.attack_techniques) if draft else [],
            cves=list(draft.cves) if draft else [],
            recommended_actions=list(draft.recommended_actions) if draft else [],
            uncertainties=uncertainties,
            model_prediction=ModelPredictionSummary(
                attack_probability=p.attack_probability,
                threshold=p.threshold,
                predicted_family=p.predicted_family,
                model_name=p.model_name,
                model_version=p.model_version,
            ),
            investigation_steps=steps,
        )
        return {"report": report, "status": "done"}

    graph: StateGraph[InvestigationState] = StateGraph(InvestigationState)
    graph.add_node("plan", plan_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "investigate")
    graph.add_conditional_edges(
        "investigate", route_after_investigate, {"critic": "critic", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "critic", route_after_critic, {"investigate": "investigate", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_investigation(
    alert: Alert,
    deps: AgentDeps,
    investigation_id: str | None = None,
    now: datetime | None = None,
) -> InvestigationResult:
    """Run one investigation end to end with a fresh usage ledger shared by both models."""
    usage = UsageTotals()
    deps.investigator.usage = usage
    deps.critic.usage = usage
    inv_id = investigation_id or uuid.uuid4().hex
    started = now or datetime.now(UTC)
    start = time.perf_counter()
    app = build_graph(deps)
    final: InvestigationState = app.invoke(
        {
            "investigation_id": inv_id,
            "alert": alert,
            "plan": [],
            "messages": [],
            "evidence": [],
            "tool_calls": [],
            "draft": None,
            "critic_issues": [],
            "all_issues": [],
            "iteration": 0,
            "tool_budget_remaining": deps.tool_budget,
            "status": "planning",
            "started_at": started,
            "errors": [],
            "report": None,
        },
        config={"recursion_limit": 50},
    )
    report = final.get("report")
    assert report is not None
    return InvestigationResult(
        investigation_id=inv_id,
        report=report,
        plan=final.get("plan", []),
        evidence=final.get("evidence", []),
        tool_calls=final.get("tool_calls", []),
        messages=final.get("messages", []),
        critic_issues=final.get("critic_issues", []),
        all_issues=final.get("all_issues", []),
        iteration=final.get("iteration", 0),
        tool_budget_remaining=final.get("tool_budget_remaining", deps.tool_budget),
        status=final.get("status", "done"),
        usage=usage,
        latency_ms=(time.perf_counter() - start) * 1000,
        errors=final.get("errors", []),
        alert_id=alert.alert_id,
        event_id=alert.event_id,
        model_investigator=deps.investigator.model,
        model_critic=deps.critic.model,
    )
