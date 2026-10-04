"""Deterministic severity rubric shared by the finalizer, the critic and the Phase 5 golden set."""

from __future__ import annotations

from typing import Any, cast

from secops.schemas.agent import ATTACK_FAMILIES, AttackFamily, Severity, Verdict

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


_FAMILY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("heartbleed", "rare_exploit"),
    ("infiltration", "rare_exploit"),
    ("ddos", "ddos"),
    ("distributed denial", "ddos"),
    ("brute", "brute_force"),
    ("patator", "brute_force"),
    ("password", "brute_force"),
    ("port_scan", "port_scan"),
    ("portscan", "port_scan"),
    ("port scan", "port_scan"),
    ("scan", "port_scan"),
    ("web_attack", "web_attack"),
    ("web attack", "web_attack"),
    ("sql", "web_attack"),
    ("xss", "web_attack"),
    ("botnet", "botnet"),
    ("bot", "botnet"),
    ("c2", "botnet"),
    ("exploit", "rare_exploit"),
    ("rare", "rare_exploit"),
    ("dos", "dos"),
    ("denial", "dos"),
    ("slowloris", "dos"),
    ("hulk", "dos"),
    ("goldeneye", "dos"),
)


def normalize_family(text: str | None) -> AttackFamily | None:
    """Map the model's free-text family to one of the Phase 1 families, or None.

    Exact names win; otherwise the first keyword match in priority order (ddos before dos,
    heartbleed before anything web). Unknown text maps to None so the caller can fall back to
    the detector's family and say so."""
    if not text:
        return None
    t = text.strip().lower()
    if t in ATTACK_FAMILIES:
        return cast(AttackFamily, t)
    for needle, family in _FAMILY_KEYWORDS:
        if needle in t:
            return cast(AttackFamily, family)
    return None


SUCCESS_MEAN_BWD_BYTES = 10_000
SUCCESS_MIN_FLOWS = 5


def success_indicator(alert_event_id: str | None, evidence: list[Any]) -> bool:
    """Deterministic success heuristic: in the get_related_events aggregate anchored on the
    alert's own event, at least 5 flows between the pair returned on average 10 KB or more per
    flow to the source (data left the host repeatedly: a Heartbleed leak, a payload download, a
    session). A brute-force burst of banners and rejects stays far below the byte bar; a single
    ordinary web response fails the flow-count bar."""
    for ev in evidence:
        if ev.tool != "get_related_events" or ev.kind != "tool_result":
            continue
        anchor = ev.payload.get("anchor") or {}
        if alert_event_id is not None and str(anchor.get("event_id")) != str(alert_event_id):
            continue
        pair = ev.payload.get("same_pair") or {}
        count = int(pair.get("count") or 0)
        if count < SUCCESS_MIN_FLOWS:
            continue
        if int(pair.get("total_bwd_bytes") or 0) / count >= SUCCESS_MEAN_BWD_BYTES:
            return True
    return False
