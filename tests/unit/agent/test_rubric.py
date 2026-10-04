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
