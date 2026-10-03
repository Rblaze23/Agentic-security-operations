import pandas as pd
import pytest
from pydantic import ValidationError

from secops.schemas.tools import AssetLookupInput, IpEnrichmentInput
from secops.tools.enrichment import EnrichmentTools, load_seeds


@pytest.fixture(scope="module")
def tools() -> EnrichmentTools:
    return EnrichmentTools(load_seeds())


def test_get_asset_known_unknown_and_invalid(tools: EnrichmentTools) -> None:
    found = tools.get_asset(AssetLookupInput(ip="192.168.10.50"))
    assert found.status == "found" and found.asset is not None
    assert found.asset.hostname == "web-server-16" and found.asset.criticality == "high"
    assert "HTTP/80" in found.asset.services and found.asset.zone == "victim_lan"
    assert found.source.startswith("https://www.unb.ca/")
    missing = tools.get_asset(AssetLookupInput(ip="10.9.9.9"))
    assert missing.status == "not_found" and missing.asset is None
    with pytest.raises(ValidationError):
        AssetLookupInput(ip="not-an-ip")
    with pytest.raises(ValidationError):
        AssetLookupInput(ip="192.168.10.50", extra=1)  # type: ignore[call-arg]


def test_enrich_ip_zones_and_known_attackers(tools: EnrichmentTools) -> None:
    kali = tools.enrich_ip(IpEnrichmentInput(ip="205.174.165.73"))
    assert kali.is_private is False and kali.zone == "attacker_external"
    assert kali.known_attacker is True and "Kali" in (kali.attacker_notes or "")
    assert kali.asset is None

    fw = tools.enrich_ip(IpEnrichmentInput(ip="172.16.0.1"))
    assert fw.is_private is True and fw.zone == "firewall" and fw.known_attacker is True
    assert "NAT" in (fw.attacker_notes or "")

    victim = tools.enrich_ip(IpEnrichmentInput(ip="192.168.10.8"))
    assert victim.is_private and victim.zone == "victim_lan" and victim.known_attacker is False
    assert victim.asset is not None and victim.asset.os.startswith("Windows Vista")

    unknown = tools.enrich_ip(IpEnrichmentInput(ip="8.8.8.8"))
    assert unknown.zone is None and unknown.known_attacker is False and unknown.asset is None
    assert unknown.source.startswith("https://")


def test_seeded_victims_occur_in_the_flows(tools: EnrichmentTools, flows: pd.DataFrame) -> None:
    """Every seeded victim must appear in the fixture flows (catches transcription typos)."""
    dsts = set(flows["Dst IP"].astype(str)) | set(flows["Src IP"].astype(str))
    for asset in tools.seeds.assets:
        if asset.zone == "victim_lan":
            assert asset.ip in dsts, asset.ip


def test_no_ground_truth_in_outputs(tools: EnrichmentTools) -> None:
    out = tools.enrich_ip(IpEnrichmentInput(ip="192.168.10.50")).model_dump()
    assert not {"label", "family", "is_attack"} & set(out)
    assert not {"label", "family", "is_attack"} & set(out["asset"])
