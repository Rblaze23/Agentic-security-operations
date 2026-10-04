import pytest

from secops.agent.rubric import severity_for


@pytest.mark.parametrize(
    ("family", "crit", "success", "verdict", "expected"),
    [
        ("port_scan", "medium", False, "true_positive", "low"),
        ("port_scan", "critical", False, "true_positive", "medium"),
        ("brute_force", "medium", False, "true_positive", "medium"),
        ("brute_force", "high", False, "true_positive", "high"),
        ("brute_force", "high", True, "true_positive", "high"),  # one bump only
        ("ddos", "high", False, "true_positive", "critical"),
        ("dos", "medium", False, "true_positive", "high"),
        ("rare_exploit", "high", False, "true_positive", "critical"),
        ("web_attack", "medium", False, "false_positive", "low"),
        ("port_scan", "low", False, "false_positive", "low"),  # clamped at low
        ("botnet", None, False, "needs_human_review", "medium"),
        (None, None, False, "false_positive", "low"),
    ],
)
def test_severity_rubric(
    family: str | None, crit: str | None, success: bool, verdict: str, expected: str
) -> None:
    assert severity_for(family, crit, success, verdict) == expected  # type: ignore[arg-type]


def test_normalize_family_maps_model_prose_to_known_families() -> None:
    from secops.agent.rubric import normalize_family

    assert normalize_family("brute_force") == "brute_force"
    assert normalize_family("Heartbleed (detector family: web_attack)") == "rare_exploit"
    assert normalize_family("DDoS flood") == "ddos"
    assert normalize_family("DoS slowloris") == "dos"
    assert normalize_family("internal port scan") == "port_scan"
    assert normalize_family("SSH password guessing") == "brute_force"
    assert normalize_family("something new") is None
    assert normalize_family(None) is None


def test_success_indicator_is_a_return_transfer_heuristic() -> None:
    from datetime import UTC, datetime

    from secops.agent.rubric import success_indicator
    from secops.schemas.agent import Evidence

    def ev(count: int, bwd: int, anchor: int = 7) -> Evidence:
        return Evidence(
            evidence_id="E1",
            tool="get_related_events",
            arguments={},
            kind="tool_result",
            summary="s",
            payload={
                "anchor": {"event_id": anchor},
                "same_pair": {"count": count, "total_bwd_bytes": bwd},
            },
            retrieved_at=datetime.now(UTC),
            untrusted_text=False,
        )

    assert success_indicator("7", [ev(19, 72_000_000)]) is True  # Heartbleed: ~3.8 MB per flow
    assert success_indicator("7", [ev(2469, 463_704)]) is False  # FTP burst: 188 B per flow
    assert success_indicator("7", [ev(1, 19_969)]) is False  # one ordinary web response
    assert success_indicator("7", [ev(19, 72_000_000, anchor=8)]) is False  # other anchor
    assert success_indicator("7", []) is False
