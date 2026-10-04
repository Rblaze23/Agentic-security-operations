"""Deterministic severity rubric shared by the finalizer, the critic and the Phase 5 golden set."""

from __future__ import annotations

from secops.schemas.agent import Severity, Verdict

LEVELS: list[Severity] = ["low", "medium", "high", "critical"]

BASE_BY_FAMILY: dict[str, Severity] = {
    "port_scan": "low",
    "brute_force": "medium",
    "web_attack": "medium",
    "botnet": "medium",
    "dos": "high",
    "ddos": "high",
    "rare_exploit": "high",
}


def _shift(level: Severity, delta: int) -> Severity:
    i = max(0, min(len(LEVELS) - 1, LEVELS.index(level) + delta))
    return LEVELS[i]


def severity_for(
    family: str | None,
    asset_criticality: str | None,
    success_indicator: bool,
    verdict: Verdict,
) -> Severity:
    """Base by family (unknown -> medium), +1 if the target asset is high/critical or a success
    indicator was observed (one bump only), -1 for a false positive; clamped to low..critical."""
    level = BASE_BY_FAMILY.get(family or "", "medium")
    if asset_criticality in ("high", "critical") or success_indicator:
        level = _shift(level, +1)
    if verdict == "false_positive":
        level = _shift(level, -1)
    return level
