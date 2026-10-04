# Agentic Security Operations Platform
ML-powered intrusion detection and evidence-grounded agentic alert investigation

> **Status: Phases 1–2 of 7 complete.** A supervised flow-level detector and an attack-family
> classifier are trained, measured, explained, registered, and served by a typed FastAPI service
> from self-contained model bundles, in a non-root container. Phases 3–7 (security tools,
> LangGraph investigation agent, evaluation harness, productionisation, portfolio polish) are not
> started. Nothing here claims to be deployed.

## What this is

A defensive security platform built end-to-end, phase by phase, with every number measured:

```
network flows ──▶ detector (LightGBM) ──▶ alert ──▶ LangGraph investigation ──▶ triage report
                                                        │ tools: event search, correlation,
                                                        │ asset/IP enrichment, CVE, ATT&CK,
                                                        │ and the detector itself
                                                        ▼
                                              evaluation + regression gate + cost tracking
```

Design: `docs/superpowers/specs/2026-10-03-platform-architecture-design.md`.
Phase 1 plan: `docs/superpowers/plans/2026-10-03-phase1-ml-detection-foundation.md`.

## Phase 1 results (measured 2026-10-03, MLflow run ids in `docs/evaluation.md`)

Dataset: CIC-IDS-2017 in the **DistriNet-corrected** version (KU Leuven), 2,099,976 flows →
1,714,956 after exact-duplicate removal, 82 flow features, identifiers never used as features.
Split: chronological within each (day, label) group, 70/15/15; threshold and champion chosen on
validation only; test scored once. Details and data-quality findings: `docs/dataset.md`.

**Binary detector** (test split, 257,252 flows, 16.5% attacks; operating point = max recall at
validation FPR ≤ 1%):

| Model | Weighting | Test PR-AUC | Recall | FPR | Precision | Fit time |
|---|---|---|---|---|---|---|
| Logistic regression | balanced | 0.9930 | 98.2% | 0.57% | 97.1% | 89 s |
| XGBoost | balanced | 0.9999 | 99.95% | 0.53% | 97.4% | 41 s |
| **LightGBM (champion)** | none | **0.9999** | **99.93%** | **0.45%** | **97.8%** | 21 s |

**Attack-family classifier** (six families, attack rows only): champion LightGBM, validation
macro-F1 0.9996, test macro-F1 0.9613, test weighted-F1 0.9989. The macro gap is one class:
`web_attack` has 16 test rows and 8 flows of other families were labelled as web attack.

**Held-out Friday (the honest number).** Train on Monday–Thursday, test on Friday, where Botnet and
DDoS never appear in training, threshold reused unchanged from the champion:

| Metric | Friday |
|---|---|
| PR-AUC | 0.9991 |
| Recall / FPR at the reused threshold | 99.3% / 1.67% |
| Recall on **Botnet** (unseen) | **6.0%** |
| Recall on DDoS (unseen) | 100% |
| Recall on Portscan | 99.2% |
| Recall if the threshold were the conventional 0.5 | 1.3% |

The in-distribution 0.9999 does not transfer: unseen command-and-control traffic is almost
entirely missed, and DDoS is caught only because the operating threshold is tiny (0.00024). This is
the measured argument for the rest of the platform: a flow-level score needs context (burst
counts, asset criticality, known-bad addresses, vulnerability data) before it is a triage decision.

**Ablations** (champion config, one variable changed): adding `Dst Port` as a feature raises test
PR-AUC to 1.0000 and cuts missed attacks from 31 to 3, confirming the port is a testbed shortcut
and justifying its exclusion. Dropping "Attempted" flows instead of relabelling them benign changes PR-AUC by less than 0.0001;
the authors' recommended policy stays the default.

Top SHAP features of the champion: Bwd Packet Length Std, Packet Length Std, Bwd Init Win Bytes,
Bwd Packet Length Mean (response-side packet statistics).

## Repository layout

```
src/secops/
  config.py            settings from environment (SECOPS_* variables)
  data/                schema, manifest + download/verify, ingest, clean, split, build, CLI
  detection/           feature spec, model factory, metrics, SHAP, plots, train, registry, bundle, CLI
  schemas/             typed contracts: flow metadata + features, prediction, alert, API envelopes
  api/                 FastAPI service: settings, API-key auth, DetectorService, routes, logging
configs/               data defaults and one YAML per experiment
scripts/               docker_smoke.py (container smoke test used by CI)
tests/                 unit (no real data), integration (986-row fixture), fixtures
docs/                  dataset.md, evaluation.md, interview-notes.md, design spec and plan
data/README.md         dataset source, terms, hashes, citations
```

## Local setup

Requirements: Python 3.12 via [uv](https://docs.astral.sh/uv/), 8 GB RAM for the full build,
`libgomp1` for LightGBM on Debian/Ubuntu, GNU make (optional; every target is a one-line `uv run`).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cp .env.example .env            # set SECOPS_DATA_DIR to a Linux-native path
make setup                      # uv sync --all-groups && pre-commit install
make test                       # ruff, mypy, unit tests, integration tests on the fixture
```

On WSL, keep the repository anywhere but put `SECOPS_DATA_DIR` and the virtual environment on the
Linux filesystem: `/mnt/*` cannot create the symlinks uv needs, so the Makefile exports
`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/secops`.

### Data and training

```bash
export SECOPS_DATA_DIR=$HOME/data/secops
make data-download              # 344 MB zip, verified by SHA-256; extracts and verifies 5 CSVs
make data-build                 # ~1 min and 5 GB RAM per policy; writes Parquet + JSON reports
make train-all                  # binary (6 runs), family (4), held-out (1), ablations (2)
make mlflow-ui                  # browse runs, artifacts, registered models
uv run secops-train report --experiment secops/detection-binary
```

Each run logs parameters, metrics (including per-label recall and recall at fixed FPRs), the
PR curve, confusion matrix, SHAP summary, feature spec, threshold, a threshold sweep with per-label
recall, a false-positive breakdown, an explainer background sample, and the model;
`secops-train promote-best` applies the documented champion rule (best validation metric, ties
within 0.0005 to the simpler model, then lower validation FPR) and moves the `champion` alias.

## Serving (Phase 2)

The champions are served by a FastAPI service from a **model bundle**: `secops-train export`
resolves the registry alias once and writes a self-contained directory (estimator, feature spec,
threshold, class order, SHAP background sample, provenance with the run's measured metrics). The
API loads bundles from `SECOPS_MODEL_DIR` and never talks to MLflow. Contract, error table and
bundle format: `docs/api.md`.

```bash
make export-models                       # -> $SECOPS_MODEL_DIR/{detector,family}
export SECOPS_API_KEYS=dev-key
make api                                 # http://localhost:8000
curl -s -H "X-API-Key: dev-key" localhost:8000/model | jq .detector.metrics.test_pr_auc
make docker-build && make compose-up     # same service in a non-root container
```

| Endpoint | What it returns |
|---|---|
| `POST /predict`, `POST /predict/batch` (≤ 1,000) | attack probability, the bundle's operating threshold, `is_alert`, attack family (alerts only), top-5 SHAP contributions, and an `Alert` object for rows above threshold |
| `GET /model` | bundle provenance and the training run's logged metrics |
| `GET /health` | readiness, open (no key) |

Requests carry flow metadata (IPs, ports, timestamp) separately from the exactly-82-name feature
dictionary; metadata is never used as a feature. Latency, one flow per request including SHAP:
in-process p50 8.0 ms / p95 9.3 ms on the fixture bundle; 15–26 ms HTTP round trip to the
container serving the real champions, whose probabilities match the in-process service exactly
(`docs/api.md`, "Measured").

## Agent tools (Phase 3)

Seven read-only, typed tools give the investigation agent real evidence to cite: event search
and neighbourhood aggregation over an event store holding all 1.7 M flows (SQLAlchemy + Alembic,
SQLite now, PostgreSQL later), asset and IP enrichment from the documented testbed, CVE lookup
from NVD with a cache and rate limiter, MITRE ATT&CK technique lookup from the official STIX
bundle (v19.2), and the detector as a tool. Every output names its source, external text is
flagged untrusted, and a registry test proves no tool exposes a ground-truth label. Details:
`docs/tools.md`; threat model: `docs/security.md`.

```bash
make load-events     # 1.7 M flows -> $SECOPS_DATA_DIR/events.db (migration applied first)
make fetch-attack    # ATT&CK STIX bundle -> technique index
```

## Investigation agent (Phase 4)

An alert becomes an evidence-grounded triage report through a LangGraph state machine:
plan → investigate → critic → finalize. Claude Opus 5.5 plans and investigates with the Phase 3
tools (budget 12 calls); a deterministic critic plus Claude Sonnet 5.5 reject any finding that
cites evidence the tools did not return; after two rejections the report is finalized as
`needs_human_review`. Severity comes from a fixed rubric, never from the model. Every model call
can be recorded and replayed, so the four real scenarios run in CI at zero cost. Details and the
measured table: `docs/agent.md`.

```bash
uv run secops-agent investigate --event-id 1110604     # score the stored flow, investigate, persist
uv run secops-agent show <investigation_id>
uv run python scripts/record_agent_scenarios.py        # dry run; --go re-records the four scenarios
```

Measured on the four recorded scenarios (2026-10-04, Opus 5.5 investigator, Sonnet 5.5 critic):

| Scenario | Event | Verdict | Severity | Family | Conf. | Tool calls | Critic rejections | LLM calls (Opus + Sonnet) | Opus input tokens (incl. cache reads) | Cache-read share | Opus output tokens | Cost | Wall-clock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ftp_bruteforce` | 1110604 | true_positive | high | brute_force | 0.95 | 5 | 0 | 4 + 1 | 25,977 | 51% | 2,607 | $0.122 | 34 s |
| `internal_portscan` | 3083227 | true_positive | low | port_scan | 0.90 | 5 | 1 | 5 + 2 | 40,082 | 44% | 4,487 | $0.210 | 50 s |
| `benign_high_score` | 357329 | needs_human_review | low | port_scan | 0.00 | 4 | 2 | 5 + 2 | 35,592 | 50% | 3,705 | $0.167 | 47 s |
| `heartbleed` | 2251110 | needs_human_review | high | heartbleed (detector family: web_attack) | 0.00 | 7 | 2 | 5 + 2 | 46,766 | 38% | 5,047 | $0.249 | 56 s |

Both true positives were found and grounded (T1110.001 and T1046 cited from the ATT&CK lookup).
The benign flow and the Heartbleed flow (CVE-2014-0160 found) ended as `needs_human_review`:
the investigator's drafts had the right verdicts, but the critic rejected them twice for
over-strict grounding objections. That is the safe failure mode, and the critic's precision is
what Phase 5 measures first. Total recorded cost $0.748.

## Security considerations

Settings and API keys come from environment variables only; `.env` is git-ignored; `gitleaks`
runs in pre-commit; dataset files and model bundles never enter git. The API authenticates every
inference route with a constant-time API-key check and fails closed when no key is configured,
refuses unauthenticated and oversized requests before reading the body, validates every field
(exact feature-name set, bounded batches, finite numbers, no unknown fields), never echoes input
values in error bodies, binds to loopback outside the container, and runs as a non-root user in a
read-only container. Model bundles are cloudpickle and are therefore a code-execution trust boundary: they
are built from the same lockfile as the image and mounted read-only. Rate limiting, persistence,
tracing and the agent's tool boundaries arrive with Phases 3–6.

## Limitations (Phase 1)

- One testbed, one week of 2017 traffic, profile-generated benign flows: numbers are not
  transferable to another network without re-measurement.
- The primary split tests the *end* of each attack burst after training on its start; it is easy by
  construction. Quote the held-out-Friday numbers for generalisation.
- Validation is used both for early stopping and for the threshold (documented optimism).
- `web_attack` (104 rows) and `rare_exploit` (47 rows) are too small for reliable per-class
  metrics; the latter is excluded from the family classifier.
- Deduplication removes scan *volume*; the detector learns the shape of a probe, not the count.

## Citations

Sharafaldin, Lashkari, Ghorbani. *Toward Generating a New Intrusion Detection Dataset and
Intrusion Traffic Characterization.* ICISSP 2018.
Liu, Engelen, Lynar, Essam, Joosen. *Error Prevalence in NIDS datasets: A Case Study on
CIC-IDS-2017 and CSE-CIC-IDS-2018.* IEEE CNS 2022.
Engelen, Rimmer, Joosen. *Troubleshooting an Intrusion Detection Dataset: the CICFlowMeter and its
flaw.* WTMC 2021.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Architecture, dataset choice, evaluation strategy | done |
| 1 | Dataset pipeline, baselines, MLflow, SHAP, registry | done |
| 2 | FastAPI detection service, typed schemas, model bundles, Docker | done |
| 3 | Security tools: event search, correlation, enrichment, CVE, ATT&CK, detector-as-tool | **done** except the detector tool (in progress) |
| 4 | LangGraph investigation agent with critic and loop limits | **done** |
| 5 | Golden set, agent evaluation harness, regression gate, cost tracking | not started |
| 6 | PostgreSQL, Langfuse, auth, CI/CD, Compose, Cloud Run | not started |
| 7 | Diagrams, demo, portfolio polish | not started |
