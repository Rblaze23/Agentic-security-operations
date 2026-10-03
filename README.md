# Agentic Security Operations Platform
ML-powered intrusion detection and evidence-grounded agentic alert investigation

> **Status: Phase 1 of 7 (ML detection foundation) complete.** A supervised flow-level detector and
> an attack-family classifier are trained, measured, explained and registered. Phases 2–7
> (detection API, security tools, LangGraph investigation agent, evaluation harness,
> productionisation, portfolio polish) are not started. Nothing here claims to be deployed.

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
| Logistic regression | balanced | 0.9930 | 98.2% | 0.57% | 97.1% | 97 s |
| XGBoost | balanced | 0.9999 | 99.95% | 0.53% | 97.4% | 134 s |
| **LightGBM (champion)** | none | **0.9999** | **99.93%** | **0.45%** | **97.8%** | 105 s |

**Attack-family classifier** (six families, attack rows only): champion LightGBM, validation
macro-F1 0.9998, test macro-F1 0.9619, test weighted-F1 0.9991. The macro gap is one class:
`web_attack` has 16 test rows and 8 DoS flows were mislabelled as web attack.

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
  detection/           feature spec, model factory, metrics, SHAP, plots, train, registry, CLI
configs/               data defaults and one YAML per experiment
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
PR curve, confusion matrix, SHAP summary, feature spec, threshold, an explainer background sample,
and the model; `secops-train promote-best` moves the `champion` alias in the MLflow registry.

## Security considerations

Phase 1 handles no secrets and exposes no network service. Settings come from environment
variables only; `.env` is git-ignored; `gitleaks` runs in pre-commit; dataset files never enter
git. The threat model for the API and the agent's tools is written with Phases 2–6.

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
| 1 | Dataset pipeline, baselines, MLflow, SHAP, registry | **done** |
| 2 | FastAPI detection service, typed schemas, Docker | not started |
| 3 | Security tools: event search, correlation, enrichment, CVE, ATT&CK, detector-as-tool | not started |
| 4 | LangGraph investigation agent with critic and loop limits | not started |
| 5 | Golden set, agent evaluation harness, regression gate, cost tracking | not started |
| 6 | PostgreSQL, Langfuse, auth, CI/CD, Compose, Cloud Run | not started |
| 7 | Diagrams, demo, portfolio polish | not started |
