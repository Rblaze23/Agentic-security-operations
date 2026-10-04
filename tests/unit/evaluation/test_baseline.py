from __future__ import annotations

from typing import Any

from secops.agent.critic import check_report
from secops.db.events_loader import event_id_for
from secops.evaluation.baseline import RuleBasedInvestigator
from secops.schemas.agent import DraftReport
from secops.tools.registry import ToolRegistry
from tests.unit.agent.test_graph import _alert


def test_baseline_ftp_bruteforce_is_true_positive(full_registry: ToolRegistry, flows: Any) -> None:
    result = RuleBasedInvestigator(full_registry).investigate(_alert(flows), investigation_id="b1")
    r = result.report
    assert [c.tool for c in result.tool_calls] == [
        "get_related_events",
        "get_asset",
        "enrich_ip",
        "lookup_attack_technique",
    ]
    assert (
        r.verdict == "true_positive" and r.attack_family == "brute_force" and r.severity == "high"
    )
    assert [t.technique_id for t in r.attack_techniques] == ["T1110"]
    ids = {e.evidence_id for e in result.evidence}
    assert all(set(f.evidence_ids) <= ids for f in r.findings if f.kind == "observed")
    assert (
        result.usage.cost_usd == 0.0 and result.status == "done" and result.investigation_id == "b1"
    )
    assert result.model_investigator == "rule-based" and result.prompt_version == "baseline-v1"


def test_baseline_quiet_internal_source_is_not_a_true_positive(
    full_registry: ToolRegistry, flows: Any
) -> None:
    alert = _alert(flows)
    row = flows[flows["label"].astype(str) == "BENIGN"].iloc[0]
    alert = alert.model_copy(
        update={
            "event_id": str(event_id_for(str(row["day"]), int(row["id"]))),
            "metadata": alert.metadata.model_copy(
                update={"source_ip": str(row["Src IP"]), "destination_ip": str(row["Dst IP"])}
            ),
        }
    )
    r = RuleBasedInvestigator(full_registry).investigate(alert).report
    assert r.verdict in ("false_positive", "needs_human_review")
    assert r.severity in ("low", "medium")


def test_baseline_never_cites_unknown_ids_or_unreturned_techniques(
    full_registry: ToolRegistry, flows: Any
) -> None:
    result = RuleBasedInvestigator(full_registry).investigate(_alert(flows))
    draft = DraftReport.model_validate(
        {
            **result.report.model_dump(
                exclude={"alert_id", "severity", "model_prediction", "investigation_steps"}
            ),
            "success_indicator": False,
        }
    )
    assert check_report(draft, {e.evidence_id: e for e in result.evidence}) == []
