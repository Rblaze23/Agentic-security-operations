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
