"""Input and output models of the agent tools. Every output names its `source`; no model here
carries ground-truth labels."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress, model_validator

MAX_WINDOW_HOURS = 24
MAX_LIMIT = 200
DEFAULT_LIMIT = 50
MAX_RELATED_WINDOW_MINUTES = 30
SAMPLE_IDS = 20
TOP_PORTS = 10


def _require_tz(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be tz-aware (UTC)")
    return value.astimezone(UTC)


class EventSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    source_ip: IPvAnyAddress | None = None
    destination_ip: IPvAnyAddress | None = None
    destination_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: int | None = Field(default=None, ge=0, le=255)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)

    @model_validator(mode="after")
    def _check_window(self) -> EventSearchInput:
        self.start = _require_tz("start", self.start)
        self.end = _require_tz("end", self.end)
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.end - self.start > timedelta(hours=MAX_WINDOW_HOURS):
            raise ValueError(f"window must be at most {MAX_WINDOW_HOURS} hours")
        return self


class EventSummary(BaseModel):
    """What a tool may say about one flow. Deliberately no label, family or features."""

    event_id: int
    timestamp: datetime
    source_ip: IPvAnyAddress
    source_port: int
    destination_ip: IPvAnyAddress
    destination_port: int
    protocol: int
    duration_ms: float
    fwd_packets: int
    bwd_packets: int
    fwd_bytes: int
    bwd_bytes: int
    syn_count: int
    fin_count: int
    rst_count: int


class EventSearchResult(BaseModel):
    events: list[EventSummary]
    total_matched: int
    truncated: bool
    source: Literal["event_store"] = "event_store"


class RelatedEventsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: int = Field(ge=0)
    window_minutes: int = Field(default=5, ge=1, le=MAX_RELATED_WINDOW_MINUTES)


class PortCount(BaseModel):
    port: int
    count: int


class Aggregate(BaseModel):
    count: int
    distinct_destination_ips: int
    distinct_destination_ports: int
    top_destination_ports: list[PortCount]
    total_fwd_bytes: int
    total_bwd_bytes: int
    first_seen: datetime | None
    last_seen: datetime | None
    sample_event_ids: list[int]


class RelatedEvents(BaseModel):
    status: Literal["found", "not_found"]
    anchor: EventSummary | None
    window_minutes: int
    same_source: Aggregate
    same_destination: Aggregate
    same_pair: Aggregate
    source: Literal["event_store"] = "event_store"


# ---- enrichment ------------------------------------------------------------------------------
class AssetLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ip: IPvAnyAddress


class Asset(BaseModel):
    ip: str
    hostname: str
    role: str
    os: str
    services: list[str] = Field(default_factory=list)
    criticality: Literal["low", "medium", "high", "critical"]
    zone: str | None
    public_ip: str | None = None
    notes: str | None = None


class AssetResult(BaseModel):
    status: Literal["found", "not_found"]
    asset: Asset | None
    source: str


class IpEnrichmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ip: IPvAnyAddress


class IpEnrichment(BaseModel):
    ip: str
    is_private: bool
    zone: str | None
    known_attacker: bool
    attacker_notes: str | None
    asset: Asset | None
    source: str


# ---- MITRE ATT&CK ----------------------------------------------------------------------------
MAX_TEXT_CHARS = 1000
MAX_KEYWORD_CHARS = 100
MAX_LOOKUP_RESULTS = 5
TECHNIQUE_ID_PATTERN = r"^[Tt]\d{4}(\.\d{3})?$"


class AttackLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technique_id: str | None = Field(default=None, pattern=TECHNIQUE_ID_PATTERN)
    keyword: str | None = Field(default=None, min_length=1, max_length=MAX_KEYWORD_CHARS)
    max_results: int = Field(default=MAX_LOOKUP_RESULTS, ge=1, le=MAX_LOOKUP_RESULTS)

    @model_validator(mode="after")
    def _one_of(self) -> AttackLookupInput:
        if (self.technique_id is None) == (self.keyword is None):
            raise ValueError("provide exactly one of technique_id or keyword")
        return self


class AttackTechnique(BaseModel):
    technique_id: str
    name: str
    tactics: list[str]
    description: str = Field(max_length=MAX_TEXT_CHARS)
    platforms: list[str]
    url: str
    version: str
    is_subtechnique: bool
    untrusted_text: Literal[True] = True


class AttackLookupResult(BaseModel):
    status: Literal["found", "not_found"]
    techniques: list[AttackTechnique]
    attack_version: str
    source: Literal["mitre-attack"] = "mitre-attack"


# ---- CVE (NVD) --------------------------------------------------------------------------------
CVE_ID_PATTERN = r"^CVE-\d{4}-\d{4,}$"
MAX_REFERENCES = 5


class CveLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cve_id: str | None = Field(default=None, pattern=CVE_ID_PATTERN)
    keyword: str | None = Field(default=None, min_length=1, max_length=MAX_KEYWORD_CHARS)
    max_results: int = Field(default=MAX_LOOKUP_RESULTS, ge=1, le=MAX_LOOKUP_RESULTS)

    @model_validator(mode="after")
    def _one_of(self) -> CveLookupInput:
        if (self.cve_id is None) == (self.keyword is None):
            raise ValueError("provide exactly one of cve_id or keyword")
        return self


class CveRecord(BaseModel):
    cve_id: str
    published: datetime
    last_modified: datetime
    cvss_v3_score: float | None
    cvss_v3_severity: str | None
    description: str = Field(max_length=MAX_TEXT_CHARS)
    references: list[str] = Field(max_length=MAX_REFERENCES)
    untrusted_text: Literal[True] = True


class CveLookupResult(BaseModel):
    status: Literal["found", "not_found", "unavailable"]
    records: list[CveRecord]
    cached: bool
    source: Literal["nvd"] = "nvd"
