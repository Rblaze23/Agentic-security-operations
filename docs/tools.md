# Agent tools

Seven read-only, typed tools give the investigation agent (Phase 4) real evidence to cite. Each is a
plain Python callable with a Pydantic input model and a Pydantic output model, registered through
`secops.tools.registry.build_registry`. No tool can write, shell out, read files, or reach the
network except the CVE tool's calls to NVD. Every output names its `source`; no output carries a
ground-truth label (a registry test walks every output schema to prove it).

## Contract

```python
ToolSpec(name, description, input_model, output_model, run, read_only=True, external_source)
spec.invoke({"...": "raw arguments as an LLM supplies them"}) -> output model
```

Inputs forbid unknown fields and bound every number and window; invalid input raises a
`pydantic.ValidationError` before the tool body runs. "No result" is a typed status
(`not_found`, `unavailable`), never an exception. Text that comes from an external source
(CVE and ATT&CK descriptions) is capped at 1,000 characters and flagged `untrusted_text: true`
so the agent prompt can treat it as content, not instructions.

## The tools

| Tool | Input (bounds) | Output | Backing data |
|---|---|---|---|
| `search_events` | tz-aware `start` < `end`, window ≤ 24 h; optional `source_ip`, `destination_ip`, `destination_port`, `protocol`; `limit` ≤ 200 | `EventSearchResult{events[], total_matched, truncated}` | event store |
| `get_related_events` | `event_id`, `window_minutes` 1–30 | `RelatedEvents{status, anchor, same_source, same_destination, same_pair}`; each aggregate: count, distinct destination IPs and ports, top 10 ports, byte totals, first/last seen, ≤ 20 sample ids; the anchor is excluded | event store |
| `get_asset` | `ip` | `AssetResult{status, asset{hostname, role, os, services, criticality, zone, public_ip, notes}}` | `data/seeds/assets.yaml` |
| `enrich_ip` | `ip` | `IpEnrichment{is_private, zone, known_attacker, attacker_notes, asset}` | `data/seeds/threat_intel.yaml` + assets |
| `lookup_attack_technique` | `technique_id` (`T1110`, `T1110.001`) **or** `keyword` ≤ 100 chars; `max_results` ≤ 5 | `AttackLookupResult{status, techniques[], attack_version}` | local ATT&CK index |
| `lookup_cve` | `cve_id` **or** `keyword` ≤ 100 chars; `max_results` ≤ 5 | `CveLookupResult{status: found/not_found/unavailable, records[], cached}` | NVD API 2.0 + cache |
| `predict_attack` | `event_ids` ≤ 100 **or** one feature dictionary | `PredictAttackResult{predictions[], missing_event_ids}` | Phase 2 `DetectorService` (pending the Phase 2 merge) |

`EventSummary` (what the event tools may say about a flow): event id, timestamp (UTC), 5-tuple,
duration, forward/backward packets and bytes, SYN/FIN/RST counts. Nothing else.

## Event store

`secops-data load-events --attempted-policy relabel_benign` applies the Alembic migration and loads
the processed Parquet into the `events` table (SQLite under `SECOPS_DATA_DIR`, PostgreSQL in
Phase 6 through `SECOPS_DATABASE_URL`). Each row holds the metadata, nine denormalised counters,
the 82-feature float32 vector, the ground truth (for evaluation only) and the split assignments.
`event_id = day_index × 1,000,000 + dataset_row_id`, because the dataset's `id` restarts at 1 in
every day file; the original id is kept in `source_row_id`. Indexes: `(source_ip, ts_us)`,
`(destination_ip, ts_us)`, `(ts_us)`, `(destination_port)`. Tools open the store with a read-only
engine (`PRAGMA query_only` on SQLite; a SELECT-only role on PostgreSQL).

Measured on the full `relabel_benign` dataset (1,714,956 flows): see "Measured" below.

## Seeds (local enrichment data)

Transcribed from the CIC-IDS-2017 page (`https://www.unb.ca/cic/datasets/ids-2017.html`,
fetched 2026-10-04), which documents the testbed: firewall 172.16.0.1 / 205.174.165.80, DNS and
domain controller 192.168.10.3, the Ubuntu 16 web server 192.168.10.50 (public 205.174.165.68),
the Ubuntu 12 server 192.168.10.51 (public 205.174.165.66, Heartbleed on port 444 in the schedule),
ten workstations in 192.168.10.0/24, and the attackers Kali 205.174.165.73 and Windows
205.174.165.69–71. Two deliberate choices:

- `172.16.0.1` is listed as a known-attacker address with a note: in the flow records the external
  attackers' traffic appears from the firewall's inside address (NAT), so flows from 172.16.0.1 to
  the victim LAN are attacker traffic entering through the firewall.
- The infiltration victim (the Vista host 192.168.10.8) is **not** marked compromised. The agent
  must discover its internal port scan from the event store; pre-marking it would make the Phase 5
  evaluation trivial.

A test asserts every seeded victim address occurs in the flows, which catches transcription typos.

## External sources

- **MITRE ATT&CK**: `scripts/fetch_attack.py` downloads the enterprise STIX bundle from
  `https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json`
  and builds `$SECOPS_DATA_DIR/attack/index.json` (technique id, name, tactics, truncated
  description, platforms, URL, version; revoked and deprecated patterns dropped). Built 2026-10-04
  from ATT&CK v19.2: 858 attack patterns, 149 revoked and 12 deprecated removed, 697 techniques,
  860 KB. The test fixture `tests/fixtures/attack/index_subset.json` holds the 16 techniques the
  Phase 1 attack families map to (T1110, T1046, T1498, T1499, T1190, T1071, T1595, T1021, T1059,
  T1105 and sub-techniques). ATT&CK content is © The MITRE Corporation under the ATT&CK Terms of
  Use.
- **NVD**: `services.nvd.nist.gov/rest/json/cves/2.0`, 5 requests per rolling 30 s without a key
  (`NVD_API_KEY` raises it to 50), 10 s timeout, results cached under
  `$SECOPS_DATA_DIR/cache/nvd` for 7 days and served stale when NVD is unreachable. Responses
  captured on 2026-10-04 by `scripts/capture_nvd_fixture.py` are the test fixtures
  (`tests/fixtures/nvd/`): CVE-2014-0160 (CVSS 3.1 7.5 HIGH), CVE-2007-6750, a nonexistent id, and
  the keyword `slowloris` (27 results). The opt-in live test (`pytest tests/network -m network`)
  passed on 2026-10-04. NVD data is public domain (NIST).

## Security boundaries

| Boundary | Enforcement |
|---|---|
| Read-only | tools hold a read-only engine; no tool has a write path; registry asserts `read_only=True` |
| Bounded inputs | windows ≤ 24 h / ≤ 30 min, `limit` ≤ 200, batches ≤ 100, keywords ≤ 100 chars, `max_results` ≤ 5; unknown fields rejected |
| No ground truth | `EventSummary` and every other output schema have no `label`, `family`, `is_attack`, split or feature-vector field (registry test) |
| External text | truncated to 1,000 characters, `untrusted_text: true`; tools never accept URLs or paths |
| Network | only the CVE tool, only to NVD, with timeout, rate limiter and cache; failures become `unavailable` |
| Provenance | every output has `source`; the ATT&CK index records bundle version and fetch date; NVD results record `cached` |

The Phase 4 agent converts tool exceptions (programming errors only) into `tool_error` evidence
and never passes tool text back as instructions.

## Measured (2026-10-04, SQLite on the Linux filesystem, 8-core WSL2)

| What | Result |
|---|---|
| `secops-data load-events` on `relabel_benign` | 1,714,956 rows in 262 s (4:25 wall), peak RSS 1.18 GB, `events.db` 1.06 GB |
| `search_events`, 1 h window + source filter, 200 random calls | p50 1.2 ms, p95 3.1 ms, max 16 ms |
| `get_related_events`, ±5 min, 200 random events | p50 32 ms, p95 510 ms, max 714 ms (median `same_source.count` 944, max 91,076) |
| `get_related_events`, ±30 min, 100 random events | p50 91 ms, p95 1.12 s, max 1.19 s |

The slow tail is the DoS/DDoS bursts: an anchor inside a flood has tens of thousands of same-source
flows in the window, and the aggregate runs three grouped scans plus the top-ports and sample
queries over them. A few calls per investigation is fine for Phase 4; if Phase 5's evaluation
runs make it a bottleneck, the first fix is a single-pass aggregate query and the second is
PostgreSQL in Phase 6.
