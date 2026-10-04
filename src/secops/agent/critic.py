"""Critic: deterministic grounding checks, then an LLM check of statement support."""

from __future__ import annotations

import json
import re
from typing import Any

from secops.agent.llm import LLM, LLMRefusalError, LLMTruncatedError
from secops.agent.prompts import load_prompt
from secops.schemas.agent import CriticIssue, CriticVerdict, DraftReport, Evidence

INSTRUCTION_PATTERNS = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|the\s+|your\s+|any\s+)?(previous|prior|above|earlier)?\s*"
    r"(instructions|rules|guidelines|prompt)|system prompt|you must now"
    r"|mark this (alert )?(as )?benign",
    re.IGNORECASE,
)


def _cited_ids(
    evidence: dict[str, Evidence], cited: list[str], tool: str, path: str, field: str
) -> set[str]:
    """Identifiers present in the *cited* evidence that came from `tool` with status found.
    Evidence the model did not cite does not count: a citation must support its own claim."""
    ids: set[str] = set()
    for eid in cited:
        ev = evidence.get(eid)
        if ev is None or ev.tool != tool or ev.kind != "tool_result":
            continue
        if ev.payload.get("status") != "found":
            continue
        for item in ev.payload.get(path, []):
            if isinstance(item, dict) and item.get(field):
                ids.add(str(item[field]).upper())
    return ids


def check_report(draft: DraftReport, evidence: dict[str, Evidence]) -> list[CriticIssue]:
    """Rules that need no model: ids exist, references came from tools, verdict is coherent,
    no instruction-like statements. Severity is not checked here because the rubric assigns it."""
    issues: list[CriticIssue] = []
    known = set(evidence)

    def unknown(ids: list[str], where: str, idx: int | None) -> None:
        missing = [i for i in ids if i not in known]
        if missing:
            issues.append(
                CriticIssue(
                    code="unknown_evidence",
                    message=f"{where} cites evidence ids that do not exist: {', '.join(missing)}",
                    finding_index=idx,
                )
            )

    for i, f in enumerate(draft.findings):
        unknown(f.evidence_ids, f"finding {i}", i)
        if INSTRUCTION_PATTERNS.search(f.statement):
            issues.append(
                CriticIssue(
                    code="instruction_in_evidence",
                    message=f"finding {i} restates an instruction-like phrase; "
                    "evidence text is data",
                    finding_index=i,
                )
            )
    for t in draft.attack_techniques:
        unknown(t.evidence_ids, f"technique {t.technique_id}", None)
    for c in draft.cves:
        unknown(c.evidence_ids, f"cve {c.cve_id}", None)
    for a in draft.recommended_actions:
        unknown(a.evidence_ids, f"action {a.action[:40]!r}", None)

    for t in draft.attack_techniques:
        supported = _cited_ids(
            evidence, t.evidence_ids, "lookup_attack_technique", "techniques", "technique_id"
        )
        if t.technique_id.upper() not in supported:
            issues.append(
                CriticIssue(
                    code="unsupported_technique",
                    message=f"{t.technique_id} is not in the cited lookup_attack_technique results",
                )
            )
    for c in draft.cves:
        supported = _cited_ids(evidence, c.evidence_ids, "lookup_cve", "records", "cve_id")
        if c.cve_id.upper() not in supported:
            issues.append(
                CriticIssue(
                    code="unsupported_cve",
                    message=f"{c.cve_id} is not in the cited lookup_cve results (found status)",
                )
            )

    kinds = [f.kind for f in draft.findings]
    if draft.verdict == "true_positive" and "observed" not in kinds:
        issues.append(
            CriticIssue(
                code="verdict_inconsistent",
                message="true_positive requires at least one observed finding with evidence",
            )
        )
    if draft.verdict == "false_positive" and "inference" not in kinds:
        issues.append(
            CriticIssue(
                code="verdict_inconsistent",
                message="false_positive requires an inference finding explaining "
                "why the alert is benign",
            )
        )
    return issues


def llm_check(llm: LLM, draft: DraftReport, evidence: dict[str, Evidence]) -> list[CriticIssue]:
    """Ask the critic model whether each observed finding follows from its cited evidence."""
    observed = [(i, f) for i, f in enumerate(draft.findings) if f.kind == "observed"]
    if not observed:
        return []
    payload: list[dict[str, Any]] = []
    for i, f in observed:
        payload.append(
            {
                "finding_index": i,
                "statement": f.statement,
                "evidence": [
                    {
                        "evidence_id": eid,
                        "tool": evidence[eid].tool,
                        "summary": evidence[eid].summary,
                    }
                    for eid in f.evidence_ids
                    if eid in evidence
                ],
            }
        )
    messages = [{"role": "user", "content": json.dumps({"findings": payload}, default=str)}]
    try:
        resp = llm.create(
            system=load_prompt("critic"), messages=messages, output_model=CriticVerdict
        )
    except (LLMRefusalError, LLMTruncatedError) as e:
        return [
            CriticIssue(
                code="unsupported_statement",
                message=f"critic model could not review the findings ({type(e).__name__})",
            )
        ]
    verdict = CriticVerdict.model_validate(resp.parsed or {"issues": []})
    valid = {i for i, _ in observed}
    out: list[CriticIssue] = []
    for issue in verdict.issues:
        if issue.finding_index is not None and issue.finding_index not in valid:
            continue
        out.append(
            CriticIssue(
                code="unsupported_statement",
                message=issue.message,
                finding_index=issue.finding_index,
            )
        )
    return out
