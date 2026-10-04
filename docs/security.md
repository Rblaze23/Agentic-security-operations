# Security

This document is the running threat model. It grows with each phase; sections are dated.

## Scope and assumptions (2026-10-04)

- The system is defensive: it classifies recorded network flows and investigates alerts. It never
  touches production networks, never executes payloads, and never acts on hosts.
- Secrets (API keys, database URLs, NVD key) come only from environment variables; `.env` is
  git-ignored and `gitleaks` runs in pre-commit and CI. No dataset, model bundle, cache, index or
  database is committed.
- The attacker of interest is anyone who can send requests to the API (Phase 2) or craft content
  the agent reads (Phase 4): flow metadata, CVE and ATT&CK descriptions, and later free text in
  reports.

## Phase 3: agent tools

**Assets at risk.** The event store (1.7 M flows with ground-truth labels used only for
evaluation), the NVD cache, the host running the agent.

**Threats and controls.**

| Threat | Control | Where |
|---|---|---|
| A tool is used to read or change data it should not (arbitrary SQL, file paths, URLs) | Tools expose fixed queries over a read-only engine; no tool accepts SQL, a path or a URL; inputs are typed, bounded and reject unknown fields | `secops.tools.*`, `db.session.make_engine(read_only=True)` |
| Ground truth leaks into the agent's evidence, making evaluation meaningless | Output models have no label, family, attack-flag, split or feature-vector field; a registry test walks every output JSON schema | `tests/unit/tools/test_registry.py` |
| Prompt injection through tool output (a CVE description that says "ignore previous instructions") | External text is truncated and flagged `untrusted_text: true`; Phase 4 renders tool results as data blocks and the critic rejects instructions found in evidence | `schemas/tools.py`, Phase 4 |
| Resource exhaustion through tools (huge windows, unbounded result lists, NVD hammering) | Windows ≤ 24 h, results ≤ 200, batches ≤ 100; NVD limiter (5/30 s), 10 s timeout, cache | `schemas/tools.py`, `tools/nvd.py` |
| Fabricated evidence (a CVE or technique that does not exist) | `lookup_cve` and `lookup_attack_technique` return `not_found` for unknown ids and only ever return records from NVD or the official ATT&CK bundle; the agent must cite ids the tools returned | `tools/cve.py`, `tools/attack.py` |
| Stale or tampered reference data | ATT&CK index records bundle version and fetch date; NVD cache entries record `stored_at` and are served stale only when NVD is down (`cached: true`) | `tools/attack.py`, `tools/nvd.py` |
| Seed data errors mislabel assets | Seeds cite their source and date; a test checks every seeded victim appears in the flows | `data/seeds/*.yaml`, `test_enrichment.py` |

**Explicitly not covered yet.** Authentication of the agent's own callers, rate limiting of
investigations, persistence of investigations and audit logs, PostgreSQL roles: Phase 6.

## Phase 2: detection API (summary)

API-key authentication checked in middleware before the body is read and again as a route
dependency (constant-time, fail closed), body size cap, strict validation that never echoes
input values, non-root read-only container, loopback binding outside the container, and
cloudpickle model bundles treated as a code-execution trust boundary (built from the same
lockfile, mounted read-only). Details in `docs/api.md`.

## Phase 4: investigation agent

**Assets at risk.** The Anthropic API key, the investigation records (contain flow metadata and
model output, no payloads), API spend.

| Threat | Control | Where |
|---|---|---|
| Prompt injection through evidence (CVE/ATT&CK text, future free text) | Evidence is rendered as data blocks with `untrusted_text`; the system prompt forbids treating evidence as instructions; the deterministic critic rejects instruction-like findings; a fixture test injects "ignore all previous instructions" and asserts the verdict is unchanged | `agent/tools.py`, `agent/critic.py`, `tests/unit/agent/test_graph.py` |
| Fabricated references (evidence ids, CVEs, techniques) | Critic rules require every cited id to exist and every CVE/technique to appear in the cited lookup evidence with status `found` | `agent/critic.py` |
| Runaway cost or loops | Tool budget (12), at most 2 critic rejections, `max_tokens` cap, per-investigation cost persisted; the CLI states cost after each run | `agent/graph.py`, `agent/tools.py`, `db/models.py` |
| Key leakage | Key read from the environment or git-ignored `.env` through `AgentSettings` (`SecretStr`), passed to the SDK client explicitly, never logged; fixtures store request bodies only (no headers) | `agent/settings.py`, `agent/llm.py` |
| Tool misuse by the model | The agent can only call the Phase 3 read-only registry; arguments are validated by the tool's Pydantic model before execution; invalid calls consume budget and return a validation error | `agent/tools.py` |
| Ground truth leaking into the investigation | The agent sees only tool outputs, which carry no labels (Phase 3 registry test); scenario fixtures are built from the same outputs | `tests/unit/tools/test_registry.py` |

## Phase 6: productionisation

| Threat | Control | Where |
|---|---|---|
| Abuse of the paid agent endpoint | API key required, per-key token bucket (429 + Retry-After), idempotent per alert, disabled entirely on the public deployment | `api/ratelimit.py`, `api/investigations.py`, `deploy/cloudrun.md` |
| Secrets in images or logs | Keys only from environment / Secret Manager; `.env` ignored; `gitleaks` in CI; fixtures hold request bodies without headers | CI `security` job |
| Vulnerable dependencies | `pip-audit` in CI against the lockfile | CI |
| Insecure code patterns | `bandit -ll` in CI; findings fixed or justified inline | CI |
| Database exposure | PostgreSQL only reachable inside the Compose network; the API uses a read-only engine for tools and a separate writable engine for investigations | `docker-compose.yml`, `db/session.py` |
| Tracing leaks | Traces carry ids, token counts, costs and verdicts, never prompts, evidence payloads or keys | `observability/__init__.py` |

Still open: TLS inside the Compose network (terminated at the edge), a shared rate-limit store
and an investigation queue across replicas, Cloud SQL for a deployed agent.
