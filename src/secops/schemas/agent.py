"""Agent contracts: evidence, findings, the triage report, critic issues, plan and usage."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EVIDENCE_ID_PATTERN = r"^E\d+$"
MAX_SUMMARY_CHARS = 600
TECHNIQUE_ID_PATTERN = r"^T\d{4}(\.\d{3})?$"
CVE_ID_PATTERN = r"^CVE-\d{4}-\d{4,}$"

Verdict = Literal["true_positive", "false_positive", "needs_human_review"]
Severity = Literal["low", "medium", "high", "critical"]
FindingKind = Literal["observed", "model_prediction", "inference"]
EvidenceKind = Literal["tool_result", "tool_error"]


class Evidence(BaseModel):
    """One tool result (or failure), referenced from findings by `evidence_id`."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    tool: str
    arguments: dict[str, Any]
    kind: EvidenceKind
    summary: str = Field(max_length=MAX_SUMMARY_CHARS)
    payload: dict[str, Any]
    retrieved_at: datetime
    untrusted_text: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)


class ToolCallRecord(BaseModel):
    tool: str
    arguments: dict[str, Any]
    evidence_id: str
    status: Literal["ok", "error", "budget_exhausted"]
    latency_ms: float


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: FindingKind
    statement: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _observed_needs_evidence(self) -> Finding:
        if self.kind == "observed" and not self.evidence_ids:
            raise ValueError("an observed finding must cite at least one evidence id")
        for e in self.evidence_ids:
            if not e.startswith("E") or not e[1:].isdigit():
                raise ValueError(f"bad evidence id {e!r}")
        return self


class TechniqueRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technique_id: str = Field(pattern=TECHNIQUE_ID_PATTERN)
    name: str
    evidence_ids: list[str] = Field(default_factory=list)


class CveRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cve_id: str = Field(pattern=CVE_ID_PATTERN)
    evidence_ids: list[str] = Field(default_factory=list)


class RecommendedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list)


class ModelPredictionSummary(BaseModel):
    attack_probability: float
    threshold: float
    predicted_family: str | None
    model_name: str
    model_version: int


ATTACK_FAMILIES = (
    "botnet",
    "brute_force",
    "ddos",
    "dos",
    "port_scan",
    "rare_exploit",
    "web_attack",
)
AttackFamily = Literal[
    "botnet", "brute_force", "ddos", "dos", "port_scan", "rare_exploit", "web_attack"
]


class TriageReport(BaseModel):
    """The agent's final output. `findings` keep observed facts, the model's prediction and the
    agent's inferences apart; every observed finding cites evidence ids that the critic verifies."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    verdict: Verdict
    attack_family: AttackFamily | None = None
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=2000)
    findings: list[Finding] = Field(default_factory=list)
    attack_techniques: list[TechniqueRef] = Field(default_factory=list)
    cves: list[CveRef] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    model_prediction: ModelPredictionSummary | None = None
    investigation_steps: list[str] = Field(default_factory=list)


class DraftReport(BaseModel):
    """What the investigator model produces (structured output); the finalizer adds the rest."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    attack_family: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=2000)
    findings: list[Finding] = Field(min_length=1)
    attack_techniques: list[TechniqueRef] = Field(default_factory=list)
    cves: list[CveRef] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    success_indicator: bool = False


class CriticIssue(BaseModel):
    code: Literal[
        "unknown_evidence",
        "unsupported_technique",
        "unsupported_cve",
        "severity_mismatch",
        "verdict_inconsistent",
        "instruction_in_evidence",
        "unsupported_statement",
        "missing_observed_finding",
    ]
    message: str
    finding_index: int | None = None


class ReviewIssue(CriticIssue):
    """Internal superset of CriticIssue. `CriticIssue` is the schema the critic model fills in
    (part of every recorded request), so outcomes the model never emits live here."""

    code: Literal[  # type: ignore[assignment]
        "unknown_evidence",
        "unsupported_technique",
        "unsupported_cve",
        "severity_mismatch",
        "verdict_inconsistent",
        "instruction_in_evidence",
        "unsupported_statement",
        "missing_observed_finding",
        "critic_unavailable",
    ]


class CriticVerdict(BaseModel):
    """Structured output of the LLM critic."""

    model_config = ConfigDict(extra="forbid")

    issues: list[CriticIssue] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[str] = Field(min_length=2, max_length=6)
    rationale: str = Field(max_length=1000)


# ---- usage and cost ---------------------------------------------------------------------------
# USD per million tokens: (input, output, cache write, cache read).
# Source: Claude API reference, model table cached 2026-09-25; https://platform.claude.com/docs
PRICES_USD_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}


class ModelUsage(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


def price_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> float:
    inp, out, cw, cr = PRICES_USD_PER_MTOK[model]  # KeyError for unknown models, on purpose
    return (
        input_tokens * inp + output_tokens * out + cache_write_tokens * cw + cache_read_tokens * cr
    ) / 1_000_000


class UsageTotals(BaseModel):
    by_model: dict[str, ModelUsage] = Field(default_factory=dict)

    @property
    def calls(self) -> int:
        return sum(m.calls for m in self.by_model.values())

    @property
    def cost_usd(self) -> float:
        return sum(m.cost_usd for m in self.by_model.values())

    @property
    def input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.by_model.values())

    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> None:
        cost = price_usd(model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
        m = self.by_model.setdefault(model, ModelUsage())
        m.calls += 1
        m.input_tokens += input_tokens
        m.output_tokens += output_tokens
        m.cache_read_tokens += cache_read_tokens
        m.cache_write_tokens += cache_write_tokens
        m.cost_usd += cost
