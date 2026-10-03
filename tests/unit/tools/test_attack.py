from pathlib import Path

import pytest
from pydantic import ValidationError

from secops.schemas.tools import MAX_TEXT_CHARS, AttackLookupInput
from secops.tools.attack import AttackIndex, AttackTools, build_index

SUBSET = Path(__file__).resolve().parents[2] / "fixtures" / "attack" / "index_subset.json"


def _stix(*patterns: dict[str, object]) -> dict[str, object]:
    return {"type": "bundle", "id": "bundle--x", "objects": list(patterns)}


def _pattern(ext_id: str, name: str, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "type": "attack-pattern",
        "id": f"attack-pattern--{ext_id}",
        "name": name,
        "description": f"Description of {name}. " * 80,
        "kill_chain_phases": [
            {"kill_chain_name": "mitre-attack", "phase_name": "credential-access"},
            {"kill_chain_name": "other-chain", "phase_name": "ignored"},
        ],
        "x_mitre_platforms": ["Linux", "Windows"],
        "x_mitre_version": "2.1",
        "x_mitre_is_subtechnique": "." in ext_id,
        "external_references": [
            {
                "source_name": "mitre-attack",
                "external_id": ext_id,
                "url": f"https://attack.mitre.org/techniques/{ext_id.replace('.', '/')}",
            },
            {"source_name": "somewhere", "url": "https://example.org"},
        ],
    }
    base.update(extra)
    return base


def test_build_index_from_synthetic_stix() -> None:
    bundle = _stix(
        _pattern("T1110", "Brute Force"),
        _pattern("T1110.001", "Password Guessing"),
        _pattern("T9999", "Old", revoked=True),
        _pattern("T9998", "Deprecated", x_mitre_deprecated=True),
        {
            "type": "x-mitre-collection",
            "name": "Enterprise ATT&CK",
            "x_mitre_version": "17.1",
            "modified": "2025-01-01T00:00:00.000Z",
        },
    )
    index = build_index(
        bundle, source_url="https://example/enterprise-attack.json", fetched="2026-10-04"
    )
    assert index.version == "17.1" and index.fetched == "2026-10-04"
    ids = [t.technique_id for t in index.techniques]
    assert ids == ["T1110", "T1110.001"]
    t = index.techniques[0]
    assert t.name == "Brute Force" and t.tactics == ["credential-access"]
    assert t.platforms == ["Linux", "Windows"] and t.url.endswith("/techniques/T1110")
    assert t.is_subtechnique is False and index.techniques[1].is_subtechnique is True
    assert len(t.description) <= MAX_TEXT_CHARS and t.untrusted_text is True


def test_index_round_trip(tmp_path: Path) -> None:
    index = build_index(
        _stix(_pattern("T1046", "Network Service Discovery")), source_url="u", fetched="d"
    )
    p = index.save(tmp_path / "index.json")
    assert AttackIndex.load(p) == index


@pytest.fixture(scope="module")
def tools() -> AttackTools:
    return AttackTools(AttackIndex.load(SUBSET))


def test_lookup_by_id_case_insensitive_and_subtechniques(tools: AttackTools) -> None:
    r = tools.lookup_attack_technique(AttackLookupInput(technique_id="t1110"))
    assert r.status == "found" and r.techniques[0].technique_id == "T1110"
    assert r.techniques[0].name == "Brute Force" and "credential-access" in r.techniques[0].tactics
    sub = tools.lookup_attack_technique(AttackLookupInput(technique_id="T1110.001"))
    assert sub.status == "found" and sub.techniques[0].is_subtechnique
    missing = tools.lookup_attack_technique(AttackLookupInput(technique_id="T0000"))
    assert missing.status == "not_found" and missing.techniques == []
    assert r.source == "mitre-attack" and r.attack_version


def test_lookup_by_keyword_ranks_name_matches_first(tools: AttackTools) -> None:
    r = tools.lookup_attack_technique(AttackLookupInput(keyword="brute force", max_results=3))
    assert r.status == "found" and 1 <= len(r.techniques) <= 3
    assert r.techniques[0].technique_id == "T1110"
    dos = tools.lookup_attack_technique(AttackLookupInput(keyword="denial of service"))
    assert {t.technique_id for t in dos.techniques} >= {"T1498", "T1499"}
    none = tools.lookup_attack_technique(AttackLookupInput(keyword="zzzz-no-such-thing"))
    assert none.status == "not_found"


def test_lookup_input_bounds() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        AttackLookupInput()
    with pytest.raises(ValidationError, match="exactly one"):
        AttackLookupInput(technique_id="T1110", keyword="x")
    with pytest.raises(ValidationError):
        AttackLookupInput(keyword="x" * 101)
    with pytest.raises(ValidationError):
        AttackLookupInput(keyword="x", max_results=6)
    with pytest.raises(ValidationError):
        AttackLookupInput(technique_id="not-a-technique")
