"""Asset inventory and IP enrichment from versioned YAML seeds (testbed documentation)."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from secops.schemas.tools import (
    Asset,
    AssetLookupInput,
    AssetResult,
    IpEnrichment,
    IpEnrichmentInput,
)
from secops.tools.base import ToolSpec

SEEDS_DIR = Path(__file__).resolve().parents[3] / "data" / "seeds"


@dataclass
class Zone:
    name: str
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network]
    description: str


@dataclass
class Seeds:
    source: str
    fetched: str
    zones: list[Zone]
    assets: list[Asset]
    known_attackers: dict[str, str] = field(default_factory=dict)

    def zone_of(self, ip: str) -> str | None:
        addr = ipaddress.ip_address(ip)
        for z in self.zones:
            if any(addr in n for n in z.networks):
                return z.name
        return None


def _networks(raw: dict[str, Any]) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    cidrs = raw.get("cidrs") or ([raw["cidr"]] if "cidr" in raw else [])
    return [ipaddress.ip_network(c) for c in cidrs]


def load_seeds(seeds_dir: Path = SEEDS_DIR) -> Seeds:
    assets_raw = yaml.safe_load((seeds_dir / "assets.yaml").read_text())
    intel_raw = yaml.safe_load((seeds_dir / "threat_intel.yaml").read_text())
    zones = [
        Zone(name=z["name"], networks=_networks(z), description=z.get("description", ""))
        for z in assets_raw["zones"]
    ]
    seeds = Seeds(
        source=str(assets_raw["source"]),
        fetched=str(assets_raw["fetched"]),
        zones=zones,
        assets=[],
        known_attackers={
            str(a["ip"]): str(a.get("notes", "")) for a in intel_raw["known_attackers"]
        },
    )
    for a in assets_raw["assets"]:
        seeds.assets.append(
            Asset(
                ip=str(a["ip"]),
                hostname=str(a["hostname"]),
                role=str(a["role"]),
                os=str(a["os"]),
                services=[str(s) for s in a.get("services", [])],
                criticality=a["criticality"],
                zone=seeds.zone_of(str(a["ip"])),
                public_ip=str(a["public_ip"]) if a.get("public_ip") else None,
                notes=a.get("notes"),
            )
        )
    return seeds


class EnrichmentTools:
    def __init__(self, seeds: Seeds) -> None:
        self.seeds = seeds
        self._by_ip = {a.ip: a for a in seeds.assets}

    def get_asset(self, inp: AssetLookupInput) -> AssetResult:
        asset = self._by_ip.get(str(inp.ip))
        return AssetResult(
            status="found" if asset else "not_found", asset=asset, source=self.seeds.source
        )

    def enrich_ip(self, inp: IpEnrichmentInput) -> IpEnrichment:
        ip = str(inp.ip)
        notes = self.seeds.known_attackers.get(ip)
        return IpEnrichment(
            ip=ip,
            is_private=ipaddress.ip_address(ip).is_private,
            zone=self.seeds.zone_of(ip),
            known_attacker=notes is not None,
            attacker_notes=notes,
            asset=self._by_ip.get(ip),
            source=self.seeds.source,
        )

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="get_asset",
                description=(
                    "Look up a host in the asset inventory by IP address: hostname, role, OS, "
                    "known services, criticality, network zone. Returns status=not_found for "
                    "addresses outside the inventory. It cannot scan hosts or see live state."
                ),
                input_model=AssetLookupInput,
                output_model=AssetResult,
                run=self.get_asset,
            ),
            ToolSpec(
                name="enrich_ip",
                description=(
                    "Enrich an IP address: private or public, network zone (victim LAN, firewall, "
                    "external attacker range), whether it is a known attacker address with notes, "
                    "and the asset record if any. Local data only; it cannot query external "
                    "reputation services."
                ),
                input_model=IpEnrichmentInput,
                output_model=IpEnrichment,
                run=self.enrich_ip,
            ),
        ]
