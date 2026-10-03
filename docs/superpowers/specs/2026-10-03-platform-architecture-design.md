# Agentic Security Operations Platform — Phase 0 Design

Status: **proposal, awaiting approval**
Date: 2026-10-03
Scope: architecture, dataset choice, ML task, agent design, evaluation methodology, repository layout, phased plan, risks. No implementation yet.

---

## 0. Repository state and environment (observed)

- Repo contains only a two-line `README.md` (modified, uncommitted). No code, no structure.
- Local machine: WSL2, Python 3.10.12, 8 cores, 8.7 GB RAM, Docker Desktop, no `uv` installed.
- Repo lives on `/mnt/d` (Windows drive mounted through DrvFs). File I/O on DrvFs is several times slower than on the Linux filesystem. **Recommendation:** keep code on `/mnt/d` but store the dataset and processed Parquet files under a Linux-native path (for example `~/data/secops`) referenced by an environment variable. 2.8 M rows of CSV on DrvFs will hurt.
- Python 3.10 is end-of-life in October 2026. Target **Python 3.12** inside Docker and via `uv` locally; keep 3.10 only as a floor if it costs nothing.

## 1. What we are building (restated so you can correct it)

An end-to-end, defensive security platform where:

1. A supervised ML detector scores network flow records and raises alerts.
2. A stateful LangGraph agent investigates each alert by querying narrowly-scoped tools (flow/event search, event correlation, asset inventory and IP enrichment, CVE lookup, MITRE ATT&CK lookup, and the ML detector itself).
3. The agent produces a structured triage report that separates **observed evidence**, **model prediction**, **agent inference**, and **recommendation**, with every claim pointing at evidence IDs.
4. Everything is measured: detection metrics (PR-AUC first), investigation metrics against a golden set, grounding/hallucination rate, tool-usage behaviour, latency, tokens and cost, run-to-run reliability, and regression versus a stored baseline.
5. It runs reproducibly (Docker Compose locally, Cloud Run as the deployment target), with MLflow for the ML side and Langfuse for the LLM/agent side, persisted in PostgreSQL.

Success criterion: a recruiter sees a real ML model in 60 seconds, a tool-using agent in 2 minutes, and evaluation/security/MLOps depth in 5 minutes. Every technology has a stated reason to exist.

Assumptions I am making (correct me):

- A1. LLM provider: Anthropic Claude through `langchain-anthropic`, selected via config so another provider can be swapped in. Default investigation model: a Sonnet-class model; a Haiku-class model for the cheap critic/judge experiments. You will need an `ANTHROPIC_API_KEY`.
- A2. Observability: Langfuse (self-hostable in Compose, open source) instead of LangSmith. One platform only.
- A3. Dashboard: Streamlit, reading from the API. Minimal time investment.
- A4. Budget: evaluation runs cost real money. Golden set stays small (30–50 alerts) and the full LLM evaluation is a manually-triggered or nightly workflow, not a per-PR check.
- A5. You will download the dataset yourself (the official source requires a form); scripts will verify checksums and build everything else.

---

## 2. Dataset analysis: CICIDS2017 vs UNSW-NB15

### 2.1 Side-by-side

| Criterion | CICIDS2017 | UNSW-NB15 |
|---|---|---|
| Producer | Canadian Institute for Cybersecurity, UNB (Sharafaldin, Lashkari, Ghorbani, ICISSP 2018) | UNSW Canberra Cyber Range Lab (Moustafa & Slay, MilCIS 2015) |
| Traffic origin | Real attack tools (Patator, slowloris, Hulk, GoldenEye, Heartbleed exploit, DVWA web attacks, Metasploit, Ares botnet, Nmap, LOIT) run by humans against a documented testbed over 5 days | IXIA PerfectStorm traffic generator replaying synthetic normal traffic plus attack behaviours drawn from a CVE library, 31 hours over two capture days |
| Size | ~2.83 M flows, 8 CSV files, ~850 MB unzipped (MachineLearningCSV) | ~2.54 M records in 4 CSVs; plus a pre-made train (175 341) / test (82 332) partition |
| Features | ~78 CICFlowMeter bidirectional flow features + label | 49 features (Argus/Bro basic, content, time, plus 12 derived connection-count features) |
| Labels | Benign + 14 attack labels (FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed, Web Brute Force, Web XSS, Web SQL Injection, Infiltration, Bot, PortScan, DDoS) | Normal + 9 categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) |
| Imbalance | ~80% benign overall; extreme tails (Heartbleed 11, Infiltration 36, SQLi 21) | Full set ~87% normal; the pre-made partition is far more balanced (~55% attack) and therefore unrepresentative |
| Time / identity context | Timestamps, source/destination IP and ports per flow; day-by-day scenario schedule with published attack windows and attacker/victim IPs | Timestamps and IPs in the full CSVs, absent from the popular partition; no scenario schedule |
| Known defects | CICFlowMeter TCP-termination bug, duplicate flows, ~10–20% mislabelled/reconstructed records, `Infinity`/`NaN` in rate features, duplicated `Fwd Header Length` column, label-leaking identifiers (Flow ID, IPs, timestamps); documented and **corrected** by KU Leuven DistriNet (Engelen, Rimmer, Joosen 2021; Liu et al. CNS 2022) and independently by Rosay et al. 2022 and Lanvin et al. 2023 | Thin documentation of what was emulated; mislabelling and simulation artefacts; `sttl`/`ct_state_ttl` act as near-perfect generator artefacts; "Generic" dominated by one attack style; attacks not tied to named CVEs or tools |
| License / terms | Free for research with mandatory citation of the ICISSP 2018 paper; official page requires a download form | Free for academic research in perpetuity; **commercial use strictly prohibited** (UNSW terms) |

Sources: UNB CIC dataset page (`https://www.unb.ca/cic/datasets/ids-2017.html`), DistriNet improved datasets (`https://intrusion-detection.distrinet-research.be/CNS2022/`), UNSW-NB15 page and terms (`https://research.unsw.edu.au/node/134656`, UNSWorks record), Rosay et al. ICISSP 2022, Lanvin et al. 2022 (HAL), arXiv 2502.06688 survey of NIDS datasets.

### 2.2 Recommendation: CICIDS2017, using the DistriNet corrected flows where possible

Reasons, in order of weight:

1. **The agent needs context, not just a feature vector.** CICIDS2017 flows carry timestamps, IPs and ports inside a documented testbed with a published attack schedule. That lets us build a real event store (flows are the events) so `search_events` and `get_related_events` return genuine neighbouring traffic, and an asset inventory tool that returns true facts about the victim network (DC at 192.168.10.3, Ubuntu web server hosting DVWA and the Heartbleed-vulnerable OpenSSL, Kali attacker at 205.174.165.73). UNSW-NB15 has no scenario narrative to investigate.
2. **Real tools and real CVEs give the enrichment tools a reason to exist.** Heartbleed is CVE-2014-0160, slowloris is CVE-2007-6750; Patator brute force maps to ATT&CK T1110, PortScan to T1046, DoS/DDoS to T1498/T1499, web attacks to T1190, Ares botnet to T1071 command-and-control. The CVE and ATT&CK tools retrieve structured facts the agent can cite. UNSW-NB15's "Exploits" and "Generic" categories cannot be grounded that way.
3. **A defensible temporal split exists.** The five-day schedule allows a chronological split and a held-out-scenario split (train on days 1–4, test on day 5), both of which are more honest than the random splits most papers use. UNSW-NB15's canonical partition is non-temporal and distribution-shifted in a way that is hard to defend.
4. **Its flaws are an asset for an interview.** The defects are documented, and a corrected variant exists. Choosing the corrected data, explaining why, and measuring the difference is exactly the "understands data quality" evidence recruiters look for.
5. **Licensing is simpler for a portfolio.** Citation-only, no commercial-use prohibition.

Costs we accept: a form-gated download, heavier preprocessing (dedupe, `Infinity` handling, column cleanup, identifier removal), extreme imbalance on rare classes (handled by family grouping, see §4), and the known label noise (mitigated by the corrected variant and documented as a limitation).

Variant choice: use the **DistriNet improved CIC-IDS-2017** CSVs (same CICFlowMeter feature space, corrected labels, bugs fixed, an extra "Attempted" attack label that we will treat per the authors' guidance). Fall back to the official `MachineLearningCSV.zip` if the improved files are unavailable; the pipeline must accept both, with the variant recorded in MLflow. We do not need the 50 GB pcaps.

UNSW-NB15 is not used. If the platform matures, cross-dataset generalisation is listed as future work, not scope.

---

## 3. Architecture

### 3.1 Data flow

```
                     ┌──────────────────────────┐
  CICIDS2017 CSVs ──▶│ ingest + clean + split    │──▶ Parquet (train/val/test) + events table
                     └──────────────────────────┘
                                 │
                                 ▼
                     ┌──────────────────────────┐   MLflow: params, metrics,
                     │ training pipeline         │──▶ artifacts, registered model
                     │ (LR, XGBoost, LightGBM)   │   SHAP global + local explainers
                     └──────────────────────────┘
                                 │  model + threshold + feature schema
                                 ▼
  flow features ──▶  POST /predict  ──▶ prediction ──▶ alert (if score ≥ threshold) ──▶ alerts table
                                                                 │
                                                                 ▼
                                             POST /investigations {alert_id}
                                                                 │
                                                                 ▼
                             ┌──────────────── LangGraph investigation ────────────────┐
                             │ plan ─▶ investigate ⟲ tools ─▶ critic ─▶ (loop ≤ 2) ─▶ finalize │
                             └───────────────────────────────────────────────────────────┘
                     tools: search_events · get_related_events · get_asset · enrich_ip
                            lookup_cve · lookup_attack_technique · predict_attack
                                                                 │
                                                                 ▼
                                   TriageReport (evidence / prediction / inference / recommendation)
                                                                 │
                     ┌───────────────────────────────────────────┼─────────────────────────┐
                     ▼                                           ▼                         ▼
             PostgreSQL (investigations,              Langfuse traces (tool calls,     Streamlit dashboard
             tool_calls, reports)                     tokens, latency, cost)
                                                                 │
                                                                 ▼
                                      evaluation harness (golden set, k repeats, regression gate)
```

### 3.2 Why an agent, and why this agent has a reason to exist

A single flow record is weak evidence. A port-scan alert on one SYN flow is only credible if the same source produced hundreds of flows to distinct ports in the last minute. A brute-force alert matters more if the destination is the domain controller than a workstation. A Heartbleed alert matters only if the target actually runs vulnerable OpenSSL. Which of these checks is relevant depends on the alert, so the investigation is conditional, multi-step, and must gather evidence before concluding. That is the argument for an agent instead of a fixed pipeline.

We will test that argument rather than assert it: the evaluation includes a **deterministic rule-based investigator** baseline (fixed query set per attack family, no LLM). The agent must beat it on triage and evidence metrics to justify its cost. If it does not, that result goes in the README.

### 3.3 Services (deliberately few)

| Service | Purpose | Runs where |
|---|---|---|
| `api` (FastAPI, one process) | `/predict`, `/alerts`, `/investigations`, `/evaluation`, `/health`, `/model`; loads the registered model in-process; runs the LangGraph agent as a background task | Compose + Cloud Run |
| `postgres` | events, alerts, investigations, tool_calls, model_predictions, evaluation_runs, models | Compose + Cloud SQL (later) |
| `mlflow` | experiment tracking and model registry (dev-time, file/SQLite backend) | Compose only |
| `langfuse` | agent tracing, token/cost accounting | Compose (self-hosted) or Langfuse cloud free tier |
| `dashboard` (Streamlit) | alert list, alert detail, evaluation dashboard | Compose + Cloud Run |

No separate "model service" microservice. The agent's `predict_attack` tool calls the detection module through the same interface the HTTP endpoint uses; in a deployed split it can point at the HTTP endpoint via config. No Redis, no vector DB, no message queue, no Kubernetes. Each would be added only when a measured need appears (documented in future work).

### 3.4 Technology choices and the reason each exists

| Choice | Why | Alternatives considered |
|---|---|---|
| LightGBM / XGBoost / Logistic Regression | Tabular flow features; gradient boosting is the proven strong baseline; LR is the honest floor | Deep tabular nets (no evidence of gain on this data), random forest (weaker calibration) |
| MLflow | Reproducible experiments and a model registry with versioned artefacts the API loads by name/alias | W&B (hosted), DVC (versioning only) |
| SHAP (TreeExplainer) | Per-alert feature attributions become part of the alert payload, so the agent and the analyst see *why* the model fired | Permutation importance (global only) |
| FastAPI + Pydantic v2 | Typed schemas shared between API, agent state, tools, DB; OpenAPI for free | Flask (no typing), Django (too heavy) |
| LangGraph | Explicit state, conditional edges, loop limits, checkpointing, and a graph you can draw in an interview | Plain function pipeline (no conditional routing), CrewAI/AutoGen (opaque multi-agent) |
| Langfuse | Open-source, self-hostable tracing with token/cost per trace; one observability platform only | LangSmith (managed only), OpenTelemetry only (no LLM-specific views) |
| PostgreSQL + SQLAlchemy 2 + Alembic | Investigations, tool calls and evaluation runs are relational and must survive restarts; migrations make the schema real | SQLite (fine for tests; used in unit tests only), document stores (no need) |
| Docker + Compose | Reproducible services; Cloud Run consumes the same images | Bare venv (not reproducible across machines) |
| GitHub Actions | PR checks (lint, type-check, unit, integration, security scan, small offline agent eval) plus a separate manual/nightly full LLM evaluation | GitLab CI (repo is on GitHub) |
| GCP Cloud Run | Stateless containers, scale-to-zero, cheap for a portfolio; Cloud SQL for Postgres | GKE (no need), VM (no autoscaling story) |
| `uv` + `pyproject.toml` | Fast, lockfile-based reproducible environments | pip-tools, Poetry |
| ruff, mypy, pytest, bandit, pip-audit, pre-commit, gitleaks | Lint, types, tests, static security, dependency vulnerabilities, secret detection | — |

---

## 4. ML task and evaluation protocol

### 4.1 Task definition

Two models, both trained from the same preprocessed data:

- **Detector (binary):** benign vs attack, per flow. Produces `attack_probability`. This is the alert source and the headline model.
- **Family classifier (multiclass):** given an attack flow, predict the attack family. Classes are grouped so every class has enough support to be measured:

| Family | Original labels |
|---|---|
| `brute_force` | FTP-Patator, SSH-Patator |
| `dos` | DoS slowloris, Slowhttptest, Hulk, GoldenEye |
| `ddos` | DDoS (LOIT) |
| `port_scan` | PortScan |
| `web_attack` | Web Brute Force, XSS, SQL Injection |
| `botnet` | Bot |
| `exploit_or_infiltration` | Heartbleed, Infiltration (tiny: reported but flagged as unreliable) |

Alert = detector score ≥ operating threshold; `predicted_attack_type` = family classifier argmax with its probability. Both go into the alert payload with SHAP top-k attributions.

### 4.2 Preprocessing (anti-leakage)

1. Load all day files, record the dataset variant and file checksums.
2. Drop identifier columns from the **feature set** but keep them as **event metadata**: Flow ID, Source IP, Destination IP, Source Port, Timestamp. `Destination Port` is also excluded from v1 features (it is a near-label in this testbed); an ablation measures what it adds.
3. Drop the duplicated `Fwd Header Length` column and constant columns; replace `Infinity` in rate features with NaN and let the trees handle it (LR gets median imputation + scaling in a pipeline fitted on train only).
4. Remove exact duplicate feature rows (documented count), because duplicates across splits inflate scores.
5. Family mapping as above; the corrected variant's "Attempted" label handled per the DistriNet guidance (treated as its own excluded/benign category; decision recorded).

### 4.3 Split strategy

- **Primary: chronological within scenario.** Sort by timestamp; per (day, label) take the first 70% of flows as train, next 15% validation, last 15% test. Every class exists in every split, and no flow in the test set precedes a flow in train. Threshold tuning uses validation only; test is touched once per final model.
- **Secondary: held-out scenario day.** Train on Monday–Thursday, test on Friday (Bot, PortScan, DDoS unseen during training). Only the binary detector is evaluated here. This measures generalisation to unseen attack behaviour and is reported separately with the explicit caveat.
- **Rejected: random stratified split.** Near-duplicate flows from the same attack burst land on both sides and produce the 99.9% F1 numbers common in papers. We document why we do not report it.

### 4.4 Metrics and headline reporting

Headline: **PR-AUC** on the positive (attack) class, then recall at a fixed false-positive-rate operating point (target FPR ≤ 1% on validation; final choice documented), precision, F1, ROC-AUC, confusion matrix, per-family precision/recall, Brier score / calibration curve. Accuracy is reported but never as the headline.

Class imbalance handling, compared in MLflow: no weighting, `scale_pos_weight` / class weights, and threshold tuning on validation. We expect threshold tuning to matter more than reweighting; we measure it.

Reproducibility: fixed seeds, pinned dependencies, hashed data manifest, one command (`make train` or `uv run train --config configs/lightgbm.yaml`) to reproduce each MLflow run.

---

## 5. Schemas (centralised in `src/secops/schemas/`)

Pydantic v2 models, shared by API, DB mapping, agent state and evaluation.

- `FlowFeatures`: the exact feature vector the model consumes (names validated against the registered model's feature schema).
- `Prediction`: `prediction_id, model_name, model_version, attack_probability, predicted_family, family_probabilities, threshold, is_alert, top_shap_features, timestamp`.
- `Alert`: `alert_id, event_id, timestamp, source_ip, destination_ip, source_port, destination_port, protocol, prediction: Prediction, key_features (subset), status`.
- `Event`: a stored flow with metadata + features + (hidden from the agent) ground-truth label.
- `Evidence`: `evidence_id, tool_name, arguments, summary, raw_ref (tool_call_id), retrieved_at`.
- `Finding`: `statement, kind: observed | model_prediction | inference, evidence_ids: list[str]`.
- `TriageReport`: `alert_id, verdict (true_positive | false_positive | needs_human_review), attack_family, severity (low|medium|high|critical), confidence, summary, findings: list[Finding], attack_techniques: list[{technique_id, name, evidence_ids}], cves: list[{cve_id, evidence_ids}], model_prediction, investigation_steps, recommended_actions: list[{action, rationale_evidence_ids}], uncertainties: list[str], tool_call_count, iterations, token_usage, cost_usd`.

Severity rubric (deterministic, documented): base by family (ddos/exploit=high, brute_force/web_attack/botnet=medium, port_scan=low) raised one level if the target asset is critical (DC, servers) or evidence shows success indicators (e.g., a successful-login-sized response after a brute-force burst), lowered one level if the verdict is false positive. The same rubric labels the golden set.

---

## 6. Agent design (LangGraph)

### 6.1 State

```python
class InvestigationState(TypedDict):
    alert: Alert
    plan: list[str]  # ordered questions to answer
    evidence: list[Evidence]  # append-only
    tool_calls: list[ToolCallRecord]
    draft: TriageReport | None
    critic_feedback: list[CriticIssue]
    iteration: int  # critic loops so far
    tool_budget_remaining: int
    status: Literal["planning", "investigating", "reviewing", "done", "failed"]
```

Checkpointed per investigation (LangGraph Postgres checkpointer) so a crashed run can be inspected and resumed.

### 6.2 Nodes and routing

```
plan ──▶ investigate ──▶ critic ──┬── approved ─────────────▶ finalize ──▶ END
                 ▲                └── rejected & iteration<2 ─┘
                 └──────────── (feedback appended) ◀──────────┘
                                   rejected & iteration==2 ──▶ finalize (needs_human_review, uncertainties filled)
```

- **plan**: LLM, structured output: 3–6 questions tailored to the alert (e.g., "Did 205.174.165.73 touch other ports on 192.168.10.50 within ±5 min?", "Is the destination asset critical?", "Does the destination run software with a CVE matching this pattern?").
- **investigate**: tool-calling loop (LangGraph `ToolNode`) with a hard tool budget (default 12 calls) and per-tool argument validation. Each result is converted to an `Evidence` record with a stable ID; the LLM only ever sees evidence IDs plus summaries, never free-form text it could misattribute. Produces a draft `TriageReport` through structured output.
- **critic**: two layers. (1) **Deterministic checks**, no LLM: every `observed` finding cites ≥1 evidence ID that exists in state; every ATT&CK/CVE ID cited came back from a tool; severity obeys the rubric; verdict/attack family consistent with findings. (2) **LLM check** (cheap model): does each observed finding's statement actually follow from the cited evidence summaries? Output: list of issues.
- **finalize**: assembles the final `TriageReport`, computes cost from token counts, persists, emits the Langfuse trace.

Loop limit: at most 2 critic rejections, then forced finalize with `needs_human_review`. Tool budget exhaustion also forces the critic step. Any tool exception becomes an `Evidence` record of kind `tool_error`, never a crash; the critic treats it as missing evidence.

### 6.3 Tool interfaces (narrow, typed, read-only)

| Tool | Signature | Backing data | Reason it exists |
|---|---|---|---|
| `search_events` | `(start, end, source_ip?, destination_ip?, destination_port?, protocol?, limit≤200) -> list[EventSummary]` | `events` table (indexed on src_ip, dst_ip, timestamp) | Establish bursts, scans, repeated attempts |
| `get_related_events` | `(alert_id, window_minutes≤30) -> RelatedEvents` (same src, same dst, same pair; counts, distinct ports, byte totals) | `events` table | Aggregates the agent would otherwise have to compute |
| `get_asset` | `(ip) -> Asset | None` (role, OS, services, criticality, network zone) | Asset inventory seeded from the documented CICIDS2017 testbed | Severity depends on what was hit |
| `enrich_ip` | `(ip) -> IpEnrichment` (internal/external, zone, known-attacker flag in the local threat-intel table, RFC1918 check) | Local table seeded from the testbed documentation; no paid API | External-vs-internal and known-bad context |
| `lookup_cve` | `(cve_id | keyword, max_results≤5) -> list[CveRecord]` | NVD API 2.0 (5 req/30 s unkeyed; key raises to 50) with on-disk cache and a committed fixture snapshot for tests/CI | Real CVE facts, never invented |
| `lookup_attack_technique` | `(technique_id | keyword) -> list[AttackTechnique]` | MITRE ATT&CK Enterprise STIX bundle (`mitre-attack/attack-stix-data`), downloaded once, indexed locally | Standard technique mapping |
| `predict_attack` | `(event_ids | features) -> list[Prediction]` | The registered detector, same code path as `/predict` | ML model as a tool: re-score neighbouring flows to confirm or dismiss a burst |

Boundaries: all tools are pure reads, parameter-validated by Pydantic, time-window and result-count capped, no shell, no filesystem, no outbound network except NVD (with timeout, retry budget and cache). Tool outputs are data, never instructions: every tool result is wrapped in a fixed schema and the system prompt states that tool content cannot change instructions (prompt-injection consideration, tested with an adversarial fixture).

### 6.4 LLM provider

Anthropic Claude via `langchain-anthropic`, model IDs in config. Structured outputs through tool-use/JSON schema. Prompt templates versioned in the repo and tagged in Langfuse so an evaluation run is tied to a prompt version.

---

## 7. Evaluation methodology

### 7.1 Golden set

30–50 alerts sampled from the **test split** across all families plus benign flows that the detector scored high (true false-positives). For each: expected verdict, expected family, expected severity (rubric), expected evidence (event IDs/aggregates that a correct investigation must surface, e.g., "≥50 distinct destination ports from the source within 60 s"), expected ATT&CK techniques, expected CVEs where applicable (Heartbleed → CVE-2014-0160), and an adversarial subset with injected text in event metadata. Stored as versioned JSON under `evaluation/golden/`.

### 7.2 Metric families

| Family | Metrics |
|---|---|
| Detection | PR-AUC, ROC-AUC, precision/recall/F1 at the operating point, FPR, per-family recall, calibration |
| Investigation | verdict accuracy, family agreement, severity exact and within-one agreement, evidence recall (expected evidence found), evidence precision (irrelevant evidence ratio) |
| Grounding | share of `observed` findings whose cited evidence exists (deterministic), share judged supported by an LLM judge, number of unsupported ATT&CK/CVE IDs (must be 0) |
| Behaviour | tool calls per investigation, unnecessary tool calls (vs the expected tool set), loop rate (hit iteration or budget cap), failure rate, wall-clock latency p50/p95 |
| Cost | input/output tokens, model, USD per investigation, USD per golden run |
| Reliability | k = 3 repeats; mean ± std per metric; flaky-case list |
| Baseline comparison | the same metrics for the deterministic rule-based investigator |

A single composite `investigation_score` (weighted mean of verdict accuracy, family agreement, severity within-one, evidence recall, grounding rate) is what the regression gate watches, with per-metric gates for grounding (no drop > 2 points) and cost (no rise > 25%).

### 7.3 Regression gate

`evaluation/baselines/<date>.json` holds the last accepted run. `secops eval compare --baseline ... --candidate ...` prints the before/after table and exits non-zero on regression. CI on PRs runs a **recorded-LLM** mode (cached responses, zero cost) for structural tests and a 5-alert smoke eval; the full k = 3 live run is a manual `workflow_dispatch` / nightly job that posts its table as a job summary.

---

## 8. Repository structure (proposed)

```
agentic-security-operations/
├── README.md                      # the portfolio front page
├── pyproject.toml                 # uv-managed; single package `secops`
├── uv.lock
├── Makefile                       # thin wrappers: setup, data, train, api, test, eval
├── Dockerfile                     # multi-stage; api and dashboard targets
├── docker-compose.yml             # api, postgres, mlflow, langfuse, dashboard
├── .env.example
├── .pre-commit-config.yaml
├── configs/                       # training + agent configs (YAML), prompt versions
├── data/README.md                 # how to obtain the dataset; manifest + checksums; nothing else committed
├── src/secops/
│   ├── config.py                  # pydantic-settings; all secrets from env
│   ├── schemas/                   # Alert, Prediction, Evidence, TriageReport, tool I/O
│   ├── data/                      # ingest, cleaning, split, feature spec
│   ├── detection/                 # training pipeline, model registry client, predictor, explainer
│   ├── tools/                     # one module per tool + registry; no LLM code here
│   ├── agent/                     # state, nodes, graph, prompts, critic checks
│   ├── api/                       # FastAPI app, routers, auth, rate limiting
│   ├── db/                        # SQLAlchemy models, session, Alembic migrations
│   ├── evaluation/                # golden set loader, metrics, runners, regression compare, rule-based baseline
│   └── observability/             # Langfuse wiring, cost tables, structured logging
├── dashboard/                     # Streamlit app (small)
├── tests/
│   ├── unit/ · integration/ · evaluation/ · security/
├── evaluation/
│   ├── golden/                    # versioned golden alerts
│   └── baselines/                 # accepted evaluation results
├── notebooks/                     # EDA only, never implementation
├── scripts/                       # download/verify data, seed assets, build ATT&CK index
├── docs/
│   ├── architecture.md · dataset.md · evaluation.md · security.md · deployment.md · interview-notes.md
│   └── superpowers/specs/         # design docs (this file)
└── .github/workflows/             # ci.yml (PR), eval.yml (manual/nightly), deploy.yml
```

Deviations from the brief and why: `src/secops/` as a single installable package instead of loose `src/` subfolders (clean imports, one build); `training/` merged into `src/secops/detection` + `configs/` so training is code, not a notebook; `notebooks/` at root for EDA only; `tests/security/` added for API security tests; `evaluation/` data directory separated from `src/secops/evaluation` code. Folders are created only when their first file lands.

---

## 9. Security posture

- API key authentication (header) with per-key rate limiting; secrets only via environment (`pydantic-settings`), `.env` ignored, `gitleaks` in pre-commit and CI.
- Strict Pydantic validation on every endpoint and every tool argument; bounded list sizes, time windows and query limits.
- Tools are read-only functions with explicit allow-listed parameters; no shell, no file access, no dynamic code.
- Prompt-injection handling: tool outputs wrapped as data; adversarial fixture in the golden set; critic catches instruction-following from evidence.
- `bandit`, `pip-audit`, `ruff` security rules in CI; container runs as non-root; health checks; no model files or data in images beyond the registered artefact.
- Threat model written in `docs/security.md`.

---

## 10. Phased plan and milestones

| Phase | Deliverable | Done when |
|---|---|---|
| 1. Data + ML baseline (≈ 1 week) | dataset docs + manifest, cleaning, splits, LR/XGBoost/LightGBM, imbalance + threshold study, MLflow runs, SHAP, `docs/dataset.md`, first real metrics in README | reproducible `make train` yields logged runs; test metrics reported with both split strategies |
| 2. Detection API (≈ 3 days) | FastAPI `/predict`, `/predict/batch`, `/health`, `/model`; schemas; model loading from registry; unit + API tests; Dockerfile | `docker compose up api` serves predictions from the registered model |
| 3. Tools (≈ 1 week) | events table + seeding, the seven tools, NVD cache + fixtures, ATT&CK index, asset inventory; tests per tool | each tool has unit tests and an integration test against seeded data |
| 4. Agent (≈ 1 week) | LangGraph graph, prompts, structured outputs, deterministic + LLM critic, loop limits, persistence of investigations, Langfuse tracing | an alert from the test split yields a grounded `TriageReport` end-to-end |
| 5. Evaluation (≈ 1 week) | golden set, metrics, rule-based baseline, k-repeat runner, regression compare, recorded-LLM CI mode | before/after table demonstrates a detected regression on an intentional change |
| 6. Productionisation (≈ 1 week) | Postgres + Alembic everywhere, auth/rate limits, security tests, CI (PR + eval + build), Compose, Cloud Run deployment docs and a real deploy | CI green; service reachable on Cloud Run with health check |
| 7. Portfolio polish (≈ 3 days) | diagrams, benchmark tables, charts, demo walkthrough, README sections 1–16, interview notes complete | README passes the 60 s / 2 min / 5 min test |

Each phase ends with a short review and no phase begins before the previous one's tests pass.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Dataset download friction (form-gated, large zips) | Blocks Phase 1 | You download once; `scripts/verify_data.py` checks checksums; pipeline accepts official or DistriNet variant |
| Label noise and CICFlowMeter artefacts | Misleading metrics | Prefer the corrected variant; document counts of dropped/relabelled rows; report both variants if feasible |
| Identifier and port leakage | Inflated scores | Explicit feature allow-list; port ablation; chronological split; duplicate removal |
| Tiny classes (Heartbleed, Infiltration, SQLi) | Unreliable per-class numbers | Family grouping; flag as unreliable in tables; never headline them |
| LLM non-determinism and cost | Flaky evals, bills | Temperature 0, k = 3 repeats with mean ± std, small golden set, recorded mode for CI, cost gate |
| NVD rate limits / outages | Tool failures | On-disk cache, committed fixture snapshot, graceful `tool_error` evidence |
| Prompt injection via event metadata | Wrong triage | Data wrapping, adversarial golden cases, critic check |
| 8.7 GB RAM and DrvFs I/O | Slow or OOM preprocessing | Store data on Linux FS, process per-file with typed columns, write Parquet; sample for notebooks |
| Scope creep ("one more tool") | Never finishing | Phase gates; every addition needs a measured justification |
| Over-claiming in README | Credibility loss | Metrics only from MLflow/evaluation artefacts; "TBD" otherwise |

---

## 12. Interview questions this design already answers

Why PR-AUC · why LightGBM over deep nets · how imbalance is handled and measured · why chronological and held-out-day splits · which columns leak and why · why an agent instead of rules (and how we test that claim) · why LangGraph · how loops are bounded · how hallucinated evidence is prevented (ID-referenced evidence + deterministic critic) · how ML and LLM are evaluated together · how cost is tracked · what happens when a tool fails · what happens at low confidence (`needs_human_review`) · how retraining and drift would work (MLflow registry alias swap, PSI on feature distributions; future work). These become `docs/interview-notes.md` entries as each phase lands.

---

## 13. Open decisions for you

1. **LLM provider**: Anthropic Claude as proposed, or another provider you already have credits for?
2. **Dataset variant**: DistriNet corrected CIC-IDS-2017 as primary (my recommendation) or the official CIC CSVs only?
3. **Dashboard**: Streamlit (proposed) or no dashboard until Phase 7?
4. **Deployment timing**: a real Cloud Run deployment in Phase 6, or documentation-only until the end?
5. **Python version**: 3.12 in Docker/uv (proposed); your local 3.10 would then be upgraded through `uv`.
