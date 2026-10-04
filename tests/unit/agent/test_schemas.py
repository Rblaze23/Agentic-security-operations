from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from secops.schemas.agent import (
    CriticIssue,
    Evidence,
    Finding,
    Plan,
    TriageReport,
    UsageTotals,
)


def _evidence(i: int, tool: str = "search_events", kind: str = "tool_result") -> Evidence:
    return Evidence(
        evidence_id=f"E{i}",
        tool=tool,
        arguments={"x": 1},
        kind=kind,  # type: ignore[arg-type]
        summary="one line",
        payload={"ok": True},
        retrieved_at=datetime.now(UTC),
        untrusted_text=False,
    )


def test_evidence_ids_follow_the_pattern() -> None:
    assert _evidence(3).evidence_id == "E3"
    with pytest.raises(ValidationError):
        Evidence(**{**_evidence(1).model_dump(), "evidence_id": "evidence-1"})
    with pytest.raises(ValidationError):
        Evidence(**{**_evidence(1).model_dump(), "summary": "x" * 601})


def test_finding_kinds_and_evidence_rule() -> None:
    f = Finding(kind="observed", statement="Many SYNs", evidence_ids=["E1", "E2"])
    assert f.evidence_ids == ["E1", "E2"]
    with pytest.raises(ValidationError):
        Finding(kind="observed", statement="Many SYNs", evidence_ids=[])  # observed needs evidence
    Finding(
        kind="inference", statement="Looks like a scan", evidence_ids=[]
    )  # inference may be bare
    with pytest.raises(ValidationError):
        Finding(kind="guess", statement="x", evidence_ids=[])  # type: ignore[arg-type]


def test_triage_report_shape_and_bounds() -> None:
    r = TriageReport(
        alert_id="a1",
        verdict="true_positive",
        attack_family="brute_force",
        severity="medium",
        confidence=0.8,
        summary="FTP brute force against the web server.",
        findings=[Finding(kind="observed", statement="x", evidence_ids=["E1"])],
        attack_techniques=[
            {"technique_id": "T1110", "name": "Brute Force", "evidence_ids": ["E2"]}
        ],
        cves=[],
        recommended_actions=[{"action": "Block 172.16.0.1", "evidence_ids": ["E1"]}],
        uncertainties=[],
    )
    assert r.verdict == "true_positive" and r.severity == "medium"
    assert r.model_dump()["attack_techniques"][0]["technique_id"] == "T1110"
    with pytest.raises(ValidationError):
        TriageReport(**{**r.model_dump(), "confidence": 1.5})
    with pytest.raises(ValidationError):
        TriageReport(**{**r.model_dump(), "severity": "urgent"})
    with pytest.raises(ValidationError):
        TriageReport(
            **{
                **r.model_dump(),
                "attack_techniques": [{"technique_id": "bogus", "name": "", "evidence_ids": []}],
            }
        )


def test_plan_and_critic_issue() -> None:
    p = Plan(
        questions=["Did the source hit other ports?", "Is the target critical?", "Known attacker?"],
        rationale="r",
    )
    assert len(p.questions) == 3
    with pytest.raises(ValidationError):
        Plan(questions=["only one"], rationale="r")
    with pytest.raises(ValidationError):
        Plan(questions=["q"] * 7, rationale="r")
    issue = CriticIssue(code="unknown_evidence", message="E9 does not exist", finding_index=0)
    assert issue.code == "unknown_evidence"


def test_usage_totals_accumulate_and_price() -> None:
    u = UsageTotals()
    u.add(
        "claude-opus-5-5",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    u.add(
        "claude-sonnet-5-5",
        input_tokens=500_000,
        output_tokens=0,
        cache_read_tokens=500_000,
        cache_write_tokens=0,
    )
    assert u.calls == 2
    assert u.cost_usd == pytest.approx(4.0 + 2.0 + 1.0 + 0.10)
    assert u.by_model["claude-opus-5-5"].output_tokens == 100_000
    with pytest.raises(KeyError):
        u.add(
            "claude-unknown",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
