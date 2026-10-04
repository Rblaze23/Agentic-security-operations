"""Rule-based investigator: a fixed query set per detector family and a handful of rules.

No model. This is the bar the agent must clear to justify its cost (spec §3.2): same tools,
same evidence records, same rubric, deterministic verdict rules."""

from __future__ import annotations

import time
import uuid
from typing import Any

from secops.agent.graph import InvestigationResult
from secops.agent.rubric import normalize_family, severity_for, success_indicator
from secops.agent.tools import ToolExecutor
from secops.evaluation.golden import EXPECTED_BY_FAMILY, EvidenceExpectation
from secops.schemas.agent import (
    CveRef,
    Evidence,
    Finding,
    ModelPredictionSummary,
    RecommendedAction,
    TechniqueRef,
    TriageReport,
    UsageTotals,
    Verdict,
)
from secops.schemas.alert import Alert
from secops.tools.registry import ToolRegistry

PROMPT_VERSION = "baseline-v1"
TECHNIQUE_BY_FAMILY = {
    "brute_force": "T1110",
    "port_scan": "T1046",
    "dos": "T1499",
    "ddos": "T1498",
    "web_attack": "T1190",
    "botnet": "T1071",
    "rare_exploit": "T1190",
}
PRIVATE_PREFIXES = (
    "10.",
    "192.168.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.2",
    "172.30.",
    "172.31.",
)


def _get(payload: dict[str, Any], path: str) -> Any:
    node: Any = payload
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


class RuleBasedInvestigator:
    """Rules (in order):
    1. Always call get_related_events(±5 min), get_asset(destination), enrich_ip(source).
    2. One reference lookup by detector family (ATT&CK technique; for rare_exploit on port 444
       the Heartbleed CVE).
    3. true_positive when the family's evidence predicate (the same one the golden set uses)
       holds or the source is a known attacker; false_positive when the source is private, not
       a known attacker, touched < 20 ports and the pair has < 5 flows; else needs_human_review.
    4. Severity from the rubric with the asset criticality and the deterministic success
       indicator; confidence is a fixed number per rule."""

    def __init__(self, registry: ToolRegistry, budget: int = 12) -> None:
        self.registry = registry
        self.budget = budget

    def investigate(self, alert: Alert, investigation_id: str | None = None) -> InvestigationResult:
        start = time.perf_counter()
        ex = ToolExecutor(self.registry, budget=self.budget)
        evidence: list[Evidence] = []
        n = 0

        def call(tool: str, args: dict[str, Any]) -> Evidence:
            nonlocal n
            n += 1
            _block, ev = ex.execute({"id": f"b{n}", "name": tool, "input": args})
            evidence.append(ev)
            return ev

        family = normalize_family(alert.prediction.predicted_family)
        src, dst = str(alert.metadata.source_ip), str(alert.metadata.destination_ip)
        related = call(
            "get_related_events", {"event_id": int(alert.event_id or 0), "window_minutes": 5}
        )
        asset = call("get_asset", {"ip": dst})
        enrich = call("enrich_ip", {"ip": src})
        lookup: Evidence | None = None
        if family == "rare_exploit" and alert.metadata.destination_port == 444:
            lookup = call("lookup_cve", {"keyword": "heartbleed"})
        elif family in TECHNIQUE_BY_FAMILY:
            lookup = call("lookup_attack_technique", {"technique_id": TECHNIQUE_BY_FAMILY[family]})

        rp = related.payload if related.kind == "tool_result" else {}
        ap = asset.payload if asset.kind == "tool_result" else {}
        ep = enrich.payload if enrich.kind == "tool_result" else {}
        known_attacker = bool(ep.get("known_attacker"))
        ports = int(_get(rp, "same_source.distinct_destination_ports") or 0)
        pair = int(_get(rp, "same_pair.count") or 0)
        predicate_ok = False
        if family in EXPECTED_BY_FAMILY and rp:
            for t, p, o, v, _d in EXPECTED_BY_FAMILY[family]["evidence"]:
                exp = EvidenceExpectation(tool=t, path=p, op=o, value=v, description=_d)
                got = _get(rp if t == "get_related_events" else ap, exp.path)
                if got is None:
                    continue
                if exp.op == "exists" or (exp.op == ">=" and got >= exp.value):
                    predicate_ok = True

        verdict: Verdict
        if predicate_ok or known_attacker:
            verdict, confidence, rule = (
                "true_positive",
                (0.8 if predicate_ok else 0.6),
                (
                    "the family's evidence predicate holds"
                    if predicate_ok
                    else "the source is a known attacker"
                ),
            )
        elif src.startswith(PRIVATE_PREFIXES) and not known_attacker and ports < 20 and pair < 5:
            verdict, confidence, rule = (
                "false_positive",
                0.7,
                "quiet internal source: few ports, few pair flows, not a known attacker",
            )
        else:
            verdict, confidence, rule = "needs_human_review", 0.3, "no rule fired"

        findings: list[Finding] = []
        if rp:
            findings.append(
                Finding(
                    kind="observed",
                    statement=(
                        f"In ±5 min the source had {_get(rp, 'same_source.count')} flows to "
                        f"{_get(rp, 'same_source.distinct_destination_ips')} hosts and "
                        f"{ports} ports; {pair} flows between the pair; the destination "
                        f"received {_get(rp, 'same_destination.count')} flows."
                    ),
                    evidence_ids=[related.evidence_id],
                )
            )
        if ap.get("status") == "found":
            a = ap["asset"]
            findings.append(
                Finding(
                    kind="observed",
                    statement=(
                        f"{dst} is {a.get('hostname')} ({a.get('role')}), criticality "
                        f"{a.get('criticality')}, zone {a.get('zone')}."
                    ),
                    evidence_ids=[asset.evidence_id],
                )
            )
        if ep:
            findings.append(
                Finding(
                    kind="observed",
                    statement=(
                        f"{src} is {'private' if ep.get('is_private') else 'public'}, "
                        f"zone {ep.get('zone')}, known attacker: {known_attacker}."
                    ),
                    evidence_ids=[enrich.evidence_id],
                )
            )
        p = alert.prediction
        findings.append(
            Finding(
                kind="model_prediction",
                statement=(
                    f"The detector scored the flow {p.attack_probability:.4f} against threshold "
                    f"{p.threshold:.6f}, family {p.predicted_family}."
                ),
                evidence_ids=[],
            )
        )
        findings.append(
            Finding(
                kind="inference",
                statement=f"Rule fired: {rule}.",
                evidence_ids=[related.evidence_id],
            )
        )

        techniques: list[TechniqueRef] = []
        cves: list[CveRef] = []
        if (
            lookup is not None
            and lookup.kind == "tool_result"
            and lookup.payload.get("status") == "found"
        ):
            if lookup.tool == "lookup_attack_technique":
                for t in lookup.payload.get("techniques", [])[:1]:
                    techniques.append(
                        TechniqueRef(
                            technique_id=t["technique_id"],
                            name=t["name"],
                            evidence_ids=[lookup.evidence_id],
                        )
                    )
            else:
                for r in lookup.payload.get("records", []):
                    if r["cve_id"] == "CVE-2014-0160":
                        cves.append(CveRef(cve_id=r["cve_id"], evidence_ids=[lookup.evidence_id]))
                # T1190 is not added here: no technique lookup returned it on this path
        criticality = _get(ap, "asset.criticality") if ap.get("status") == "found" else None
        severity = severity_for(
            family, criticality, success_indicator(alert.event_id, evidence), verdict
        )
        uncertainties = [
            f"{e.tool} ({e.evidence_id}) failed: {e.summary}"
            for e in evidence
            if e.kind == "tool_error"
        ]
        actions = (
            [
                RecommendedAction(
                    action=f"Review the destination host and the source's recent flows ({rule}).",
                    evidence_ids=[related.evidence_id],
                )
            ]
            if verdict != "false_positive"
            else []
        )
        report = TriageReport(
            alert_id=alert.alert_id,
            verdict=verdict,
            attack_family=family if verdict != "false_positive" else None,
            severity=severity,
            confidence=confidence,
            summary=f"Rule-based triage: {verdict} ({rule}).",
            findings=findings,
            attack_techniques=techniques,
            cves=cves,
            recommended_actions=actions,
            uncertainties=uncertainties,
            model_prediction=ModelPredictionSummary(
                attack_probability=p.attack_probability,
                threshold=p.threshold,
                predicted_family=p.predicted_family,
                model_name=p.model_name,
                model_version=p.model_version,
            ),
            investigation_steps=[f"{r.tool} -> {r.evidence_id} [{r.status}]" for r in ex.records],
        )
        return InvestigationResult(
            investigation_id=investigation_id or uuid.uuid4().hex,
            report=report,
            plan=[rule],
            evidence=evidence,
            tool_calls=list(ex.records),
            messages=[],
            critic_issues=[],
            all_issues=[],
            iteration=0,
            tool_budget_remaining=ex.remaining,
            status="done",
            usage=UsageTotals(),
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_version=PROMPT_VERSION,
            alert_id=alert.alert_id,
            event_id=alert.event_id,
            model_investigator="rule-based",
            model_critic="none",
        )
