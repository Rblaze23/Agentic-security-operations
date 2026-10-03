"""MITRE ATT&CK technique lookup over a compact index built from the enterprise STIX bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from secops.schemas.tools import (
    MAX_TEXT_CHARS,
    AttackLookupInput,
    AttackLookupResult,
    AttackTechnique,
)
from secops.tools.base import ToolSpec

ATTACK_BUNDLE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)


class AttackIndex(BaseModel):
    version: str
    fetched: str
    source_url: str
    techniques: list[AttackTechnique]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1))
        return path

    @classmethod
    def load(cls, path: Path) -> AttackIndex:
        return cls.model_validate_json(path.read_text())

    def subset(self, technique_ids: list[str]) -> AttackIndex:
        wanted = {t.upper() for t in technique_ids}
        return self.model_copy(
            update={"techniques": [t for t in self.techniques if t.technique_id in wanted]}
        )


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_TEXT_CHARS else text[: MAX_TEXT_CHARS - 1] + "…"


def build_index(bundle: dict[str, Any], source_url: str, fetched: str) -> AttackIndex:
    objects = bundle.get("objects", [])
    version = "unknown"
    for o in objects:
        if o.get("type") == "x-mitre-collection":
            version = str(o.get("x_mitre_version", version))
    techniques: list[AttackTechnique] = []
    for o in objects:
        if o.get("type") != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        ref = next(
            (r for r in o.get("external_references", []) if r.get("source_name") == "mitre-attack"),
            None,
        )
        if ref is None or not ref.get("external_id"):
            continue
        tactics = [
            p["phase_name"]
            for p in o.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack"
        ]
        techniques.append(
            AttackTechnique(
                technique_id=str(ref["external_id"]),
                name=str(o.get("name", "")),
                tactics=tactics,
                description=_truncate(str(o.get("description", ""))),
                platforms=[str(p) for p in o.get("x_mitre_platforms", [])],
                url=str(ref.get("url", "")),
                version=str(o.get("x_mitre_version", "")),
                is_subtechnique=bool(o.get("x_mitre_is_subtechnique", False)),
            )
        )
    techniques.sort(key=lambda t: t.technique_id)
    return AttackIndex(
        version=version, fetched=fetched, source_url=source_url, techniques=techniques
    )


class AttackTools:
    def __init__(self, index: AttackIndex) -> None:
        self.index = index
        self._by_id = {t.technique_id.upper(): t for t in index.techniques}

    def lookup_attack_technique(self, inp: AttackLookupInput) -> AttackLookupResult:
        if inp.technique_id is not None:
            hit = self._by_id.get(inp.technique_id.upper())
            return AttackLookupResult(
                status="found" if hit else "not_found",
                techniques=[hit] if hit else [],
                attack_version=self.index.version,
            )
        kw = (inp.keyword or "").lower()
        name_hits = [t for t in self.index.techniques if kw in t.name.lower()]
        desc_hits = [
            t for t in self.index.techniques if t not in name_hits and kw in t.description.lower()
        ]
        hits = (name_hits + desc_hits)[: inp.max_results]
        return AttackLookupResult(
            status="found" if hits else "not_found",
            techniques=hits,
            attack_version=self.index.version,
        )

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="lookup_attack_technique",
                description=(
                    "Look up MITRE ATT&CK enterprise techniques by exact id (e.g. T1110 or "
                    "T1110.001) or by keyword over names and descriptions (name matches first). "
                    "Returns at most 5 techniques with tactics, platforms, a truncated description "
                    "and the attack.mitre.org URL, from a local copy of the official STIX bundle "
                    "(version reported). Descriptions are external text. It cannot map a flow to a "
                    "technique by itself."
                ),
                input_model=AttackLookupInput,
                output_model=AttackLookupResult,
                run=self.lookup_attack_technique,
                external_source="mitre-attack",
            ),
        ]
