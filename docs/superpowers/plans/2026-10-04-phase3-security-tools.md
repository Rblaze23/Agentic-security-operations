# Phase 3 — Security Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Git policy for this project: the user owns all commits and pushes. Every task ends by reporting the change set (files touched, suggested commit message); never run `git commit`, `git merge` or `git push`.**

**Goal:** Give the Phase 4 agent seven narrowly scoped, read-only, typed tools with real data behind them: event search and correlation over the full flow dataset, asset and IP enrichment from the documented testbed, CVE lookup from NVD, MITRE ATT&CK technique lookup from the official STIX bundle, and the Phase 1 detector as a tool. Every tool is framework-agnostic (plain Python with Pydantic input and output models), bounded in what it can touch and how much it can return, and covered by tests that run without network or full data.

**Architecture:** A new `secops.tools` package holds one module per tool plus a registry (`ToolSpec`: name, description, input model, output model, callable, limits) that Phase 4 converts into LangGraph tools. An **event store** (SQLAlchemy 2 + Alembic, SQLite locally, PostgreSQL in Phase 6 with the same code) holds every flow of the processed dataset with its metadata, a hidden ground-truth label and the float32 feature vector, indexed by time and IP; a loader fills it from the Phase 1 Parquet. Enrichment data is seeded from the CIC-IDS-2017 testbed description (asset inventory, zones, known attacker addresses) as versioned YAML. The CVE tool wraps NVD API 2.0 with an on-disk cache, rate limiting and an injectable fetcher; a committed fixture snapshot serves tests. The ATT&CK tool reads a compact index built once from the enterprise STIX bundle; a committed fixture subset serves tests. The detector tool calls Phase 2's `DetectorService` in-process, by feature vector or by event id.

**Tech Stack:** SQLAlchemy 2, Alembic, SQLite (dev/tests) → PostgreSQL (Phase 6), Pydantic v2, httpx (NVD client), PyYAML, pytest. No LLM code in this phase.

**Spec:** `docs/superpowers/specs/2026-10-03-platform-architecture-design.md` sections 3.2 (why an agent), 6.3 (tool interfaces and boundaries), 9 (security); project brief sections 8 (tools), 9 (evidence-grounded reasoning), 13 (tool permission boundaries), 15 (database, migrations). Consumes Phase 1 `FeatureSpec`, processed Parquet columns (`id`, `day`, metadata, 82 features, `label_raw`, `label`, `family`, `is_attack`, split columns) and Phase 2 `DetectorService`, `Prediction`, `PredictRequest`.

## Global Constraints

- Tools are **read-only**: no shell, no filesystem writes except the CVE cache under `SECOPS_DATA_DIR`, no outbound network except NVD (and the one-time ATT&CK download script). Every parameter is validated and bounded (time windows, result counts, batch sizes).
- Tool outputs are **data**, returned as Pydantic models; free text from external sources (CVE and ATT&CK descriptions) is truncated to a documented length and marked by source so the agent prompt can treat it as untrusted content.
- Ground-truth labels in the event store are **never** returned by a tool (they exist for the Phase 5 golden set only); the `EventSummary` schema has no label field and a test proves it.
- Event timestamps are UTC; all tool time inputs are tz-aware or rejected.
- No data, caches, databases or indexes in git; fixtures only (≤ 1,000 events, a handful of CVEs and techniques).
- Measured numbers only (`TBD` until measured); every seed fact about the testbed cites its source line in `docs/tools.md`.
- The user owns git: tasks report their change set; nothing is committed by the executor.

## Review Focus

1. **A `search_events` call with a window longer than the cap, `end < start`, a naive datetime, or `limit` above 200** must be rejected by the input model, not clamped silently. → Task 2 `test_search_events_input_bounds`.
2. **`get_related_events` on an unknown event id** returns a typed "not found" result (not an exception leaking a SQL message), and on a known id never includes the anchor event in its own counts. → Task 2 `test_related_events_unknown_and_self_exclusion`.
3. **Ground truth must not leak**: no tool output model has a `label`, `family` or `is_attack` field, enforced by a test over the registry. → Task 7 `test_no_tool_output_exposes_ground_truth`.
4. **An NVD outage or rate-limit response** yields a typed `CveLookupResult` with `status="unavailable"` and the cached entries if any, never an exception or an invented CVE; a CVE id that does not exist yields `status="not_found"`. → Task 5 `test_cve_lookup_unavailable_and_not_found`.
5. **`predict_attack` with more than 100 ids, or ids that do not exist**, is rejected or reported per id, and a request by raw features goes through exactly the same `FeatureSpec` validation as the API. → Task 6 `test_predict_attack_bounds_and_missing_ids`.

---

# Part A — Design

## A1. Why these tools, and why real data behind them

The Phase 1 held-out result is the motivation: a flow-level score catches 6% of unseen botnet traffic and would catch 1.3% of Friday's attacks at a conventional threshold. Turning a score into a triage decision needs context the model cannot see: how many flows the same source produced in the last minute and to how many ports (`get_related_events`), what the destination is and whether it matters (`get_asset`), whether the source is a known external attacker or an internal host (`enrich_ip`), whether the targeted service has a vulnerability that matches the pattern (`lookup_cve`), the standard name for the behaviour (`lookup_attack_technique`), and whether the neighbouring flows are also scored as attacks (`predict_attack`). Each tool answers one such question from a real source so that every claim in the Phase 4 report can cite an `evidence_id`.

## A2. Event store

**Why a database now.** `search_events` and `get_related_events` need indexed lookups by IP and time over 1.7 M flows, hundreds of times per evaluation run. Scanning Parquet with pandas per query costs about half a second each; an indexed table costs milliseconds. The brief requires migrations and names PostgreSQL for persistence; SQLAlchemy 2 with Alembic gives the same code on SQLite today and PostgreSQL in Phase 6, where the alert and investigation tables join this one.

Table `events` (initial Alembic migration):

| Column | Type | Notes |
|---|---|---|
| `event_id` | BigInteger PK | the dataset `id` |
| `day` | String(10) | |
| `timestamp` | DateTime(timezone=True) | indexed |
| `source_ip`, `destination_ip` | String(45) | indexed with timestamp |
| `source_port`, `destination_port` | Integer | `destination_port` indexed |
| `protocol` | SmallInteger | |
| `flow_duration_us`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`, `syn_count`, `fin_count`, `rst_count`, `ack_count` | Integer/BigInteger | denormalised from the feature vector for summaries |
| `feature_spec_version` | String(32) | |
| `features` | LargeBinary | float32 vector in spec order (328 bytes) |
| `label_raw`, `label`, `family`, `is_attack` | String / SmallInteger | **ground truth; never exposed by tools** |
| `split_chrono`, `split_heldout` | String(5) | for golden-set construction |

Indexes: `(source_ip, timestamp)`, `(destination_ip, timestamp)`, `(timestamp)`, `(destination_port)`.

Loader: `secops-data load-events [--policy relabel_benign] [--days ...]` reads the processed Parquet in chunks of 50,000 rows and bulk-inserts; idempotent (truncate-and-load per policy, recorded in a `load_runs` row with row count, manifest digest and git sha). Expected size: TBD (measured in Task 1; the vector alone is ~560 MB).

Connection: `SECOPS_DATABASE_URL` (default `sqlite:///$SECOPS_DATA_DIR/events.db`). Tests use `sqlite:///:memory:` or a temp file loaded from the 986-row fixture through the same loader.

## A3. Tool contract

```python
class ToolSpec(BaseModel):
    name: str  # snake_case, stable; the agent calls tools by this name
    description: str  # one paragraph the LLM sees
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    run: Callable[[BaseModel], BaseModel]
    read_only: Literal[True] = True
    external_source: str | None  # "nvd" | "mitre-attack" | None (local)
```

Rules: inputs validated by Pydantic with `extra="forbid"`; every list output capped; every output carries `source` (where the data came from) and, when text is included from an external source, `untrusted_text: True`. Tools never raise for "no result": they return a typed status. Tools raise only for programming errors (which the agent layer converts to `tool_error` evidence in Phase 4).

## A4. The seven tools

| Tool | Input (bounds) | Output | Backing |
|---|---|---|---|
| `search_events` | `start`, `end` (tz-aware; `end > start`; window ≤ 24 h), optional `source_ip`, `destination_ip`, `destination_port`, `protocol`, `limit` ≤ 200 (default 50) | `EventSearchResult{events: list[EventSummary], total_matched, truncated}` | event store |
| `get_related_events` | `event_id`, `window_minutes` ≤ 30 (default 5) | `RelatedEvents{anchor, same_source: Aggregate, same_destination: Aggregate, same_pair: Aggregate, status}` where `Aggregate{count, distinct_destination_ips, distinct_destination_ports, top_destination_ports[≤10], total_fwd_bytes, total_bwd_bytes, first_seen, last_seen, sample_event_ids[≤20]}`; anchor excluded from aggregates | event store |
| `get_asset` | `ip` | `AssetResult{status, asset: Asset | None}`; `Asset{ip, hostname, role, os, services[], criticality, zone, source}` | `data/seeds/assets.yaml` from the CIC testbed table |
| `enrich_ip` | `ip` | `IpEnrichment{ip, is_private, zone, known_attacker, attacker_notes, asset: Asset | None, source}` | `data/seeds/threat_intel.yaml` + assets |
| `lookup_cve` | `cve_id` **or** `keyword` (one required; keyword ≤ 100 chars), `max_results` ≤ 5 | `CveLookupResult{status: found | not_found | unavailable, records: list[CveRecord], source, cached}`; `CveRecord{cve_id, published, last_modified, cvss_v3_score, cvss_v3_severity, description[≤ 1,000 chars], references[≤ 5], untrusted_text=True}` | NVD API 2.0 + cache |
| `lookup_attack_technique` | `technique_id` **or** `keyword`, `max_results` ≤ 5 | `AttackLookupResult{status, techniques: list[AttackTechnique]}`; `AttackTechnique{technique_id, name, tactics[], description[≤ 1,000 chars], platforms[], url, version, untrusted_text=True}` | local ATT&CK index |
| `predict_attack` | `event_ids` (≤ 100) **or** `features` (one `PredictRequest`-style dict) | `PredictAttackResult{predictions: list[Prediction], missing_event_ids[]}` | Phase 2 `DetectorService` + event store |

`EventSummary` fields: `event_id, timestamp, source_ip, source_port, destination_ip, destination_port, protocol, duration_ms, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes, syn_count, fin_count, rst_count`. No label.

## A5. Seeds and external sources

- **Assets and zones**: the CIC-IDS-2017 page documents the testbed (firewall 205.174.165.80 / 172.16.0.1, DNS and domain controller 192.168.10.3, Ubuntu web server 192.168.10.50 with public address 205.174.165.68, Ubuntu 12 server 192.168.10.51 / 205.174.165.66, workstations on Ubuntu 14.4/16.4, Windows 7/8.1/Vista/10 and macOS in 192.168.10.0/24, attackers Kali 205.174.165.73 and Windows 205.174.165.69–71). Task 3 fetches the page, transcribes the table into `data/seeds/assets.yaml` with the URL and date, and records the dataset subtlety that external attacker traffic appears in flows from `172.16.0.1` (the firewall's inside address), which `threat_intel.yaml` therefore lists as "attacker traffic via firewall NAT".
- **NVD**: `https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=…` or `?keywordSearch=…&resultsPerPage=5`; 5 requests per rolling 30 s without a key, 50 with `NVD_API_KEY`; cache under `$SECOPS_DATA_DIR/cache/nvd/<sha1 of query>.json` with a 7-day TTL; fixture snapshot in `tests/fixtures/nvd/` for CVE-2014-0160 (Heartbleed), CVE-2007-6750 (slowloris) and one keyword query, captured once by a script with the capture date recorded.
- **ATT&CK**: `scripts/fetch_attack.py` downloads the enterprise STIX bundle from the `mitre-attack/attack-stix-data` repository (URL and version recorded in the index), builds `$SECOPS_DATA_DIR/attack/index.json` (techniques: id, name, tactics, description, platforms, url, revoked/deprecated filtered out) and prints counts; `tests/fixtures/attack/index_subset.json` holds the techniques the Phase 1 attack families map to (T1110 brute force, T1046 network service discovery, T1498 and T1499 denial of service, T1190 exploit public-facing application, T1071 application-layer protocol, T1595 active scanning, T1021 remote services, T1059 command and scripting interpreter).

## A6. Security boundaries

Read-only functions; bounded parameters; the event store engine is opened read-only for tools (SQLite `?mode=ro`; a Postgres role with SELECT only in Phase 6); the CVE client has a 10 s timeout, two retries with backoff, and a token-bucket limiter; no tool accepts a URL or a path; external text is length-capped and flagged. These boundaries are listed in `docs/tools.md` and `docs/security.md` (new, threat model section for tools) and tested where testable.

## A7. Testing

Per tool: unit tests on a fixture event store (loaded from the 986-row fixture), YAML seeds, fixture NVD responses through the injectable fetcher, the ATT&CK subset index, and fixture bundles. Property-style checks: bounds, no label leakage, aggregates exclude the anchor, cache hits avoid the fetcher, rate limiter spacing. Integration: loader on the fixture end-to-end; `predict_attack` by id equals the API's prediction for the same flow. Full-data (manual, recorded in docs): load time, database size, p50/p95 latency of each event-store tool over 200 random calls.

## A8. Milestones

| # | Deliverable | Task |
|---|---|---|
| M1 | Event store: model, migration, loader, CLI, read-only session | 1 |
| M2 | `search_events`, `get_related_events`, tool base + registry skeleton | 2 |
| M3 | Seeds + `get_asset`, `enrich_ip` | 3 |
| M4 | ATT&CK fetch script, index, `lookup_attack_technique` | 4 |
| M5 | NVD client, cache, limiter, `lookup_cve` | 5 |
| M6 | `predict_attack` | 6 |
| M7 | Registry completeness tests, `docs/tools.md`, `docs/security.md`, interview notes, full-data load and latency measurements | 7 |

Definition of done: all seven tools in the registry with input/output models and tests; `secops-data load-events` loaded the full `relabel_benign` dataset with measured time and size; measured tool latencies in `docs/tools.md`; ATT&CK index built from the live bundle (version recorded); NVD fixture captured with date; no label field reachable through any tool; `make test` green; change set reported for the user to commit.

## A9. Risks

| Risk | Mitigation |
|---|---|
| SQLite write throughput for 1.7 M rows | executemany in 50k chunks inside one transaction, indexes created after load; measure; fall back to `--days` subsets for dev |
| CIC testbed table transcription errors | transcribe with the source URL and date, keep the YAML small, cross-check IPs against the flows (Task 3 asserts every seeded victim IP appears as a `Dst IP` in the fixture or full data) |
| NVD unavailable during development | fixture snapshot + cache; live test is opt-in (`-m network`) |
| ATT&CK bundle is large (~40 MB) and changes | one-time download outside tests; index records bundle version and date; fixture subset for CI |
| Tool interfaces drift from what Phase 4 needs | registry test asserts each spec has docstring, bounded ints, `source`, and JSON-schema-able models now |
| DrvFs drops mid-task | no commits by the executor anyway; the user commits at task boundaries |

---

# Part B — Tasks

Branch: the user chooses (suggested `phase-3/security-tools` from `main` after merging Phase 2). Each task: tests first, run to see them fail, implement, run green, lint and type-check, **report the change set**.

### Task 1: Event store (model, migration, loader, CLI)

**Files:** `src/secops/db/{__init__,models,session,events_loader}.py`, `alembic.ini`, `src/secops/db/migrations/{env.py,script.py.mako,versions/0001_events.py}`, `src/secops/data/cli.py` (add `load-events`), `src/secops/config.py` (`database_url`), `pyproject.toml` (sqlalchemy, alembic), `tests/unit/db/test_events_loader.py`, `tests/unit/db/test_migration.py`

**Interfaces produced:** `Settings.database_url: str | None` + `resolved_database_url()`; `db.models.Event` (SQLAlchemy mapped class) and `LoadRun`; `db.session.make_engine(url, read_only=False)`, `session_scope(engine)`; `events_loader.load_events(parquet_path, engine, feature_spec, chunk_size=50_000) -> LoadReport{rows, seconds, db_bytes}`; `events_loader.unpack_features(blob) -> np.ndarray`; Alembic `upgrade head` creates the schema; CLI `secops-data load-events --policy relabel_benign [--days monday,...]`.

Tests: migration creates `events` with the expected columns and indexes on a temp SQLite; loader on the fixture Parquet (built by `build()` in the test) inserts 986 rows, round-trips a feature vector to float32 equality, is idempotent (second load replaces), records a `load_runs` row; read-only engine refuses writes.

### Task 2: `search_events`, `get_related_events`, tool base and registry skeleton

**Files:** `src/secops/tools/{__init__,base,registry,events}.py`, `src/secops/schemas/tools.py` (tool I/O models: `EventSummary`, `EventSearchInput`, `EventSearchResult`, `RelatedEventsInput`, `RelatedEvents`, `Aggregate`), `tests/unit/tools/{__init__,conftest,test_events}.py`

**Interfaces produced:** `ToolSpec`, `Tool` protocol, `registry.get_tool(name)`, `registry.all_tools()`; `EventStoreTools(engine)` exposing `search_events(EventSearchInput) -> EventSearchResult` and `get_related_events(RelatedEventsInput) -> RelatedEvents`; `MAX_WINDOW_HOURS = 24`, `MAX_LIMIT = 200`, `MAX_RELATED_WINDOW_MINUTES = 30`.

Tests: bounds (Review Focus 1); filters by each field; `truncated` and `total_matched` semantics; unknown id (Review Focus 2); anchor excluded; aggregate counts hand-checked against pandas on the fixture; distinct ports top-10 ordering; no `label` in any output.

### Task 3: Asset inventory and IP enrichment

**Files:** `data/seeds/assets.yaml`, `data/seeds/threat_intel.yaml`, `src/secops/tools/enrichment.py`, `src/secops/schemas/tools.py` (add `Asset`, `AssetResult`, `IpEnrichment`), `tests/unit/tools/test_enrichment.py`, `docs/tools.md` (seed sources section)

Steps include fetching the CIC testbed page, transcribing it with URL and date, and asserting seeded victim IPs occur in the fixture flows. Tools: `get_asset(ip)`, `enrich_ip(ip)` (RFC1918 check via `ipaddress`, zone from the seeds, known-attacker flag, asset join).

### Task 4: MITRE ATT&CK index and lookup

**Files:** `scripts/fetch_attack.py`, `src/secops/tools/attack.py`, `src/secops/schemas/tools.py` (add `AttackTechnique`, `AttackLookupInput`, `AttackLookupResult`), `tests/fixtures/attack/index_subset.json`, `tests/unit/tools/test_attack.py`

Index builder parses STIX `attack-pattern` objects (external id from `external_references` with `source_name == "mitre-attack"`, kill-chain phases → tactics, `x_mitre_platforms`, `revoked`/`x_mitre_deprecated` filtered), stores bundle version and fetch date. Lookup by id (exact, including sub-techniques `T1110.001`) or keyword (case-insensitive over name and description, ranked by name match first). Tests on the subset; the builder is tested on a tiny synthetic STIX document.

### Task 5: NVD CVE lookup

**Files:** `src/secops/tools/nvd.py` (client: fetcher protocol, cache, limiter, parser), `src/secops/tools/cve.py` (tool), `src/secops/schemas/tools.py` (add `CveRecord`, `CveLookupInput`, `CveLookupResult`), `tests/fixtures/nvd/{cve-2014-0160.json,cve-2007-6750.json,keyword-slowloris.json,README.md}`, `scripts/capture_nvd_fixture.py`, `tests/unit/tools/test_cve.py`, `tests/network/test_nvd_live.py` (marker `network`, skipped by default)

Client: `NvdClient(fetcher, cache_dir, ttl_days=7, requests_per_30s=5 or 50 with key)`; parser maps NVD 2.0 JSON to `CveRecord` (CVSS v3.1 then v3.0 then none; English description; references capped). Tool statuses per Review Focus 4. Tests: fixture fetcher; cache hit skips fetcher; limiter spacing with a fake clock; unavailable and not-found paths; description truncation.

### Task 6: `predict_attack`

**Files:** `src/secops/tools/detector.py`, `src/secops/schemas/tools.py` (add `PredictAttackInput`, `PredictAttackResult`), `tests/unit/tools/test_detector_tool.py`

`DetectorTool(service: DetectorService, engine)`: by `event_ids` loads vectors from the store (missing ids reported), builds `PredictRequest`s with the stored metadata, calls `service.predict`; by `features` validates through the same path as the API. Review Focus 5 tests; equality with the API path on fixture rows.

### Task 7: Registry completeness, docs, full-data measurements

**Files:** `src/secops/tools/registry.py` (register all seven), `tests/unit/tools/test_registry.py`, `docs/tools.md`, `docs/security.md`, `docs/interview-notes.md`, `README.md`, `Makefile` (`load-events`, `fetch-attack`)

Tests: every tool registered, JSON schemas generate, bounded ints have `le`, no output model exposes `label`/`family`/`is_attack` (Review Focus 3), every description mentions what the tool cannot do. Full-data: `secops-data load-events` on `relabel_benign` (time, size), ATT&CK index from the live bundle (version, technique count), 200-call latency samples for the two event-store tools, recorded in `docs/tools.md`. Change set reported.

---

## Self-review notes

- Spec coverage: brief §8 tools (all six named plus the detector) → Tasks 2–6; §9 evidence IDs → every output carries `event_id`s / `cve_id`s / `technique_id`s the agent can cite; §13 boundaries → A6 and registry tests; §15 migrations → Task 1 Alembic. Deferred by design: PostgreSQL itself (Phase 6, same code), LangGraph wrapping (Phase 4), golden-set construction (Phase 5).
- Open decisions for the user: (1) SQLite now with Alembic, PostgreSQL in Phase 6 (recommended) versus PostgreSQL immediately in Compose; (2) API-key-less NVD with the 5-requests-per-30-s limit (recommended; an `NVD_API_KEY` is optional) versus requiring a key.
- Placeholders: none beyond measured-value `TBD`s; detailed test and implementation code is written at execution time following the Phase 1–2 pattern.
