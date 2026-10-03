# Phase 1 — ML Detection Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, leakage-aware supervised detector (binary attack/benign) and attack-family classifier on the DistriNet-corrected CIC-IDS-2017 flows, with every experiment tracked in MLflow, every metric measured, and the trained model registered so Phase 2 can serve it.

**Architecture:** A single installable package `secops` with a `data` subpackage (manifest verification, ingest, cleaning, splitting, a `build` CLI that writes one Parquet file with split-assignment columns) and a `detection` subpackage (feature spec, model factory, metrics and threshold selection, MLflow training CLI, SHAP explainer, registry helpers). Data and MLflow state live under a Linux-native data directory, never on the Windows-mounted repo path. Configuration is YAML for experiments and environment variables for paths and secrets.

**Tech Stack:** Python 3.12, uv, pandas + pyarrow, scikit-learn, XGBoost, LightGBM, SHAP, MLflow (SQLite backend, local artifact store), Pydantic v2 + pydantic-settings, Typer CLI, matplotlib, pytest, ruff, mypy, pre-commit, GitHub Actions (lint + unit tests only in this phase).

**Spec:** `docs/superpowers/specs/2026-10-03-platform-architecture-design.md` (Phase 0 design, approved 2026-10-03 with decisions: Anthropic Claude later, DistriNet corrected dataset, no dashboard until Phase 7, real Cloud Run in Phase 6, Python 3.12 + uv + Docker).

## Global Constraints

- Python `>=3.12`; environment managed by `uv` with a committed `uv.lock`.
- No dataset files, MLflow runs, models or `.env` committed to git. Data root is `SECOPS_DATA_DIR` (default `~/data/secops`), never under `/mnt/d`.
- Never report a metric that was not produced by an MLflow run or a saved report file. Unmeasured values are written as `TBD`.
- Accuracy is never the headline metric. Headline is PR-AUC on the attack class, then recall at the chosen false-positive-rate operating point.
- Identifier columns (`id`, `Flow ID`, `Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Timestamp`) are never model features. `Dst Port` is an explicit ablation only.
- "Attempted" flows are never a separate model label (dataset authors' directive). Default policy: relabel to BENIGN; `drop` is an ablation.
- Every split is chronological; no random shuffling across time. Validation is used for threshold selection and model choice; the test split is scored once per final configuration.
- Fixed seeds everywhere (`SECOPS_RANDOM_SEED`, default 42). Each MLflow run records the data manifest hash, cleaning policy, split strategy and feature-spec version.
- Commit after every task with a conventional-commit message ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

Inputs the spec implies but that are easy to get wrong. Each has a pinned test in the owning task.

1. **A raw CSV whose header differs from the 91 expected columns** (wrong dataset variant, or the official CIC CSVs) must fail loudly at ingest, naming the missing/unexpected columns, not silently train on misaligned features. → Task 4 `test_read_day_rejects_wrong_header`.
2. **A flow row with `Infinity` in `Flow Bytes/s`** must become NaN for tree models and a finite imputed value for logistic regression; it must never raise or be dropped silently. → Task 5 `test_clean_replaces_inf_with_nan`, Task 8 `test_logreg_pipeline_handles_nan`.
3. **A split where any test-row timestamp precedes a train-row timestamp inside the same (day, label) group** is a leak and must be rejected by a check that runs in the build pipeline. → Task 6 `test_check_no_time_leak_raises`.
4. **A validation set in which no threshold meets the FPR target** (for example a tiny sample) must return the most conservative threshold with a recorded `method` of `fallback_max_threshold`, not crash or silently pick 0.5. → Task 7 `test_select_threshold_fallback_when_no_threshold_meets_fpr`.
5. **A feature matrix with columns in a different order or with a missing column** passed to the registered model must be rejected by `FeatureSpec.to_matrix`, because Phase 2 will feed JSON payloads built independently of pandas column order. → Task 8 `test_feature_spec_rejects_reordered_columns`, `test_to_matrix_rejects_missing_column`.

---

# Part A — Phase 1 plan (the nineteen questions)

## A1. Exact dataset and why

**Dataset:** CIC-IDS-2017, **improved/corrected version** published by the DistriNet research group, KU Leuven, as supplementary material to Liu, Engelen, Lynar, Essam, Joosen, *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018*, IEEE CNS 2022, building on Engelen, Rimmer, Joosen, *Troubleshooting an Intrusion Detection Dataset: the CICFlowMeter and its flaw*, WTMC 2021.

- Download page: `https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html`
- File: `https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip` (343,549,013 bytes; server timestamp 2023-04-27). Openly served; no form.
- Measured SHA-256 of the zip on 2026-10-03: `97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883`
- Labelling code: `https://github.com/GintsEngelen/CNS2022_Code`; flow extractor: `https://github.com/GintsEngelen/CICFlowMeter` (fork with corrected TCP termination on mutual FIN, RST handling, PSH/URG counting, non-absolute Active/Idle times, UTC microsecond timestamps, ICMP and RST features).
- Original dataset and terms: `https://www.unb.ca/cic/datasets/ids-2017.html` (Sharafaldin, Lashkari, Ghorbani, ICISSP 2018). CIC requires citation of that paper; the DistriNet repository requests citation of the CNS 2022 paper. Neither page states a formal license text and the DistriNet code README shows no license. **We treat the data as research-use with mandatory citation of both papers, do not redistribute it, and keep it out of git.** Recorded in `data/README.md`.

Three layers are kept distinct in all documentation:

| Layer | What it is | Who did it |
|---|---|---|
| Original CIC-IDS-2017 | 5 days of pcaps, flows extracted with CICFlowMeter v3, labels from a time-and-IP schedule; ~2.83 M flows, 78 features | CIC/UNB, 2017 |
| DistriNet improved | Same pcaps re-extracted with the fixed CICFlowMeter fork, relabelled with published per-attack rules, "Attempted" sub-labels added; 2,099,976 flows, 82 features + 9 identifier/label columns | KU Leuven DistriNet, 2021–2023 |
| Our preprocessing (this plan) | Identifier removal, infinity handling, Attempted policy, exact-duplicate removal, family grouping, chronological splitting, Parquet build | this repository |

Why this variant: the Phase 0 reasons (real tools, real CVEs, timestamps and IPs for an event store, defensible temporal splits) hold for both; the corrected variant additionally fixes the flow-extraction bug that split flows on the first FIN, removes the absolute-timestamp encoding in Active/Idle features (a leakage channel), relabels mis-timed attacks, and publishes every labelling rule. The official CSVs have a different header and can be run through the same pipeline with a second manifest; that is a documented future ablation, not a Phase 1 deliverable.

## A2. Acquisition and reproducibility procedure

1. `make data-download` runs `uv run secops-data download`: fetches the zip into `$SECOPS_DATA_DIR/raw/`, verifies the zip SHA-256, extracts the five CSVs, verifies each CSV's SHA-256 and byte size, and writes `raw/MANIFEST_VERIFIED.json`.
2. `make data-build` runs `uv run secops-data build --attempted-policy relabel_benign` and `--attempted-policy drop`: ingest, clean, split, write `processed/<policy>/flows.parquet` and `reports/<policy>/{cleaning_report,split_report}.json`.
3. Every MLflow run is tagged with the manifest digest and the git commit, so any number traces back to exact bytes and code.
4. If the DistriNet server disappears, the hashes let anyone verify a copy obtained elsewhere; the manifest refuses anything else.

Measured manifest (2026-10-03):

| File | Bytes | Rows (excl. header) | SHA-256 |
|---|---|---|---|
| monday.csv | 207,875,155 | 371,624 | `51fe5dc962626efb4ae70dce0303072fb780da0932822b651202ee9c2fbc1aff` |
| tuesday.csv | 178,397,720 | 322,078 | `e2a0a5b631dfc6b455cc9f9a88b944110637d70a7f74171473925f76f38b6b0c` |
| wednesday.csv | 291,290,505 | 496,641 | `bf46c5f3c792e8817381f724511229569606918eaf07ac986d7a2592b6341bc2` |
| thursday.csv | 189,519,159 | 362,076 | `78a4d11eaf473d099e30e71ddb01e0f38218e844c0a9cdd36602145d674af482` |
| friday.csv | 285,188,226 | 547,557 | `ebd499e6f23bd59f9cb81bec28178491b02b925fa5640a24215c9437d79482d0` |

Total: 2,099,976 flows.

## A3. Dataset structure and target labels

**Columns (91):** `id` (row index), `Flow ID`, `Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Protocol`, `Timestamp` (UTC with microseconds; verified: FTP-Patator runs 12:19–13:20 UTC = 9:19–10:20 ADT, matching the published schedule), 81 flow statistics from `Flow Duration` to `Total TCP Flow Time`, `Label`, `Attempted Category` (-1 = not attempted; 0–6 = reason code).

**Feature set (82):** `Protocol` plus the 81 statistics, including the fork's new `Fwd RST Flags`, `Bwd RST Flags`, `ICMP Code`, `ICMP Type`, `Total TCP Flow Time`. All numeric. No column is constant across all five files.

**Measured label distribution (all days):**

| Label | Rows | Attempted rows (separate label in file) |
|---|---|---|
| BENIGN | 1,582,566 | — |
| Portscan | 159,066 | 0 |
| DoS Hulk | 158,468 | 581 |
| DDoS | 95,144 | 0 |
| Infiltration - Portscan | 71,767 | 0 |
| DoS GoldenEye | 7,567 | 80 |
| FTP-Patator | 3,972 | 12 |
| DoS Slowloris | 3,859 | 1,847 |
| SSH-Patator | 2,961 | 27 |
| DoS Slowhttptest | 1,740 | 3,368 |
| Botnet | 736 | 4,067 |
| Web Attack - Brute Force | 73 | 1,292 |
| Infiltration | 36 | 45 |
| Web Attack - XSS | 18 | 655 |
| Web Attack - SQL Injection | 13 | 5 |
| Heartbleed | 11 | 0 |

Attempted rows total 11,979. After the default policy (relabel to BENIGN): 1,594,545 benign and 505,431 attack rows (24.07% positive) **before** duplicate removal. Post-deduplication counts: TBD (produced by the build report in Task 6).

**Targets:**

- Binary `is_attack`: 1 for any non-BENIGN label after the Attempted policy.
- Family `family` (attack rows only):

| Family | Labels | Rows before dedup |
|---|---|---|
| `dos` | DoS Hulk, GoldenEye, Slowloris, Slowhttptest | 171,634 |
| `port_scan` | Portscan, Infiltration - Portscan | 230,833 |
| `ddos` | DDoS | 95,144 |
| `brute_force` | FTP-Patator, SSH-Patator | 6,933 |
| `botnet` | Botnet | 736 |
| `web_attack` | Web Brute Force, XSS, SQL Injection | 104 |
| `rare_exploit` | Heartbleed, Infiltration | 47 |

`rare_exploit` (47 rows) is **excluded from family-classifier training and scoring** and reported qualitatively; it stays in the binary task as attack. `web_attack` (104 rows) is trained but flagged low-support. `Infiltration - Portscan` maps to `port_scan` because the flows are an internal host's port sweep; that the source is internal is context the Phase 3 enrichment tool surfaces, not something flow statistics encode. Original labels are preserved in `label_raw`.

Change from Phase 0: the family previously named `exploit_or_infiltration` becomes `rare_exploit` and is excluded from the classifier, because the corrected labels leave too few rows to train or evaluate.

## A4. Data-quality issues discovered (measured on the improved CSVs)

1. **Exact duplicate feature+label rows: 312,810 (14.9%)**, concentrated in Friday (177,187) and Thursday (81,261); overwhelmingly identical port-scan probes. Genuine traffic, but copies on both sides of a split inflate metrics and over-weight port scans. Policy: drop exact duplicates on `FEATURE_COLS + label` before splitting; log per-label counts before and after. The Phase 3 event store keeps every row.
2. **Infinity values: 8 cells** across `Flow Bytes/s` and `Flow Packets/s` (zero-duration flows). Replace with NaN; trees handle NaN natively, logistic regression gets median imputation fitted on train only.
3. **No NaN cells, no constant columns, no non-numeric feature columns** (the original CSVs had a duplicated `Fwd Header Length` column and NaNs; the fork removed them).
4. **Attempted flows (11,979)**: attacker-generated flows with no malicious payload (closed port, startup/teardown, target unresponsive, mis-implemented attack). The authors state: "Under no circumstance should the 'Attempted' flows be treated as a separate label for your Machine Learning model" and recommend relabelling them as benign. Default `relabel_benign`; ablation `drop`. Recorded as MLflow tag `attempted_policy`.
5. **Corrected-label consequences**: web attacks shrink from ~2,180 (original) to 104 confirmed-malicious flows; Botnet has 736 live flows and 4,067 post-shutdown connection attempts; Infiltration is 36 flows plus a 71,767-flow internal port scan the original labelled benign; DoS Hulk in the original used `Connection: close`, so the fork relabels by packet signature rather than by time window alone. Documented in `docs/dataset.md` with the authors' per-attack notes.
6. **Benign traffic is shared across days**: every day carries benign background from the same hosts; Monday is benign-only. The chronological within-group split therefore keeps benign from every day in every split.
7. **Timestamps are reliable** (UTC, microseconds, monotone per day), which the original CSVs were not. This is what makes the chronological split possible.

## A5. Leakage risks and controls

| Risk | Channel | Control |
|---|---|---|
| Identifier leakage | `Src IP`, `Dst IP`, `Flow ID` identify attacker (172.16.0.1, 192.168.10.8) and victim (192.168.10.50) | Never features; metadata only |
| Port leakage | Attack `Dst Port` is one of {21, 22, 80, 444, 8080} in this testbed | Excluded from v1 features; measured as ablation `v1-withport` and reported as a caveat |
| Temporal leakage | Random split puts neighbouring flows of one burst on both sides | Chronological split within (day, label); leak check asserts no val/test timestamp precedes train within a group |
| Duplicate leakage | Identical rows across splits | Exact-duplicate removal before split |
| Preprocessing leakage | Imputer/scaler fitted on all data | Fitted inside the sklearn `Pipeline` on train only |
| Threshold leakage | Threshold tuned on test | Tuned on validation; test scored once |
| Label-encoded time | Original CICFlowMeter encoded absolute timestamps into Active/Idle features | Fixed by the fork; build report asserts Active/Idle maxima are below one day |
| Row-order leakage | `id` correlates with time and day | Never a feature |

## A6. Train/validation/test split

**Primary: `chrono_within_group`.** For each group (day, label-after-policy), sort by `Timestamp` then `id`; first 70% → `train`, next 15% → `val`, last 15% → `test`. Every class in every split; no future flow in train; benign from each day in each split; attack bursts split by time so the test set is the *end* of each attack. Realistic for "the model has seen the start of a campaign".

**Secondary: `heldout_friday`.** Train and validation from Monday–Thursday (chronological within group, 85/15), test = all of Friday (Botnet, DDoS, Portscan unseen as labels; the internal `Infiltration - Portscan` from Thursday is a related behaviour). Binary task only. Measures generalisation to unseen attack behaviour and is reported with that caveat.

**Rejected: random stratified split**, documented in `docs/dataset.md` with the leakage argument. Never run, so no inflated number is ever printed.

Both assignments are columns (`split_chrono`, `split_heldout`) in one processed Parquet, so one build serves every experiment.

## A7. Preprocessing pipeline

```
raw CSV (per day) ──read_day──▶ typed DataFrame (float32 features, UTC timestamps, 'day' column)
   │  header must equal RAW_COLUMNS (91) exactly
   ▼
concat ──clean(policy)──▶ label_raw kept; label = base label or BENIGN (policy);
   │                       family, is_attack derived; ±inf → NaN; exact duplicates dropped
   ▼
assign_chrono_within_group ─▶ split_chrono ; assign_heldout_day ─▶ split_heldout
   │   check_no_time_leak on both
   ▼
processed/<policy>/flows.parquet  +  reports/<policy>/{cleaning_report,split_report}.json
```

Model-side preprocessing lives inside each estimator: logistic regression = `SimpleImputer(median) → StandardScaler → LogisticRegression`; XGBoost/LightGBM consume NaN directly. No target encoding, no resampling, no feature selection in v1.

## A8. Class-imbalance strategy

Binary positive rate is ~24% before dedup (TBD after); the family task is extreme (`web_attack` 104 vs `port_scan` 230,833).

Compared in MLflow, per model: `weighting=none`; `weighting=balanced` (sklearn `compute_sample_weight("balanced")` passed as `sample_weight` to all three models); threshold tuning on validation (always applied; the main lever for the binary task). No SMOTE or other resampling in v1: it invents flow rows and is unreliable on flow statistics; documented as considered-and-rejected.

## A9. Baseline models

| Model | Role | Config |
|---|---|---|
| Logistic Regression | honest floor, calibrated linear baseline | `C=1.0`, `max_iter=2000`, `solver=lbfgs`, imputer + scaler pipeline |
| XGBoost | strong tree baseline | `tree_method=hist`, `n_estimators=600`, `learning_rate=0.05`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping 50 rounds on validation `aucpr` |
| LightGBM | expected champion, fastest | `n_estimators=1000`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping 50 rounds on validation `average_precision` |

No hyper-parameter search in Phase 1 beyond early stopping; an Optuna study is future work. "Expected champion" is a hypothesis, not a result.

## A10. Metrics

**Binary (headline first):** PR-AUC (average precision) on the attack class; recall, precision, F1 and FPR at the selected threshold; ROC-AUC; Brier score; confusion matrix; accuracy (reported, never headline); recall at FPR ∈ {0.1%, 0.5%, 1%, 5%}; **per-original-label recall** at the threshold (did the detector catch the 11 Heartbleed flows?), so rare attacks are not hidden behind the aggregate.

**Family (multiclass):** macro-F1 (headline), weighted-F1, per-class precision/recall/F1/support, confusion matrix, log-loss.

**Held-out-day:** the binary metrics on Friday with the chrono-split threshold reused unchanged (a deployed threshold is not re-tuned on the future).

All metrics computed on `val` and `test`, logged with `val_` and `test_` prefixes.

## A11. Threshold selection

On the validation split, among thresholds whose FPR ≤ `max_fpr` (default 0.01), choose the one with the highest recall; ties → the higher threshold. Also record the max-F1 threshold for comparison. If no threshold satisfies the constraint, return the strictest threshold and set `method="fallback_max_threshold"`. The chosen threshold is saved as `threshold.json` next to the model and becomes the registered model's `is_alert` rule in Phase 2.

Interview note: in a SOC the budget is analyst capacity, so the false-positive rate, not F1, governs the operating point.

## A12. Explainability plan

- Global: SHAP `TreeExplainer` on a 20,000-row sample of the validation split for tree models; mean |SHAP| per feature as CSV and a bar PNG logged to MLflow. For logistic regression, `LinearExplainer` on the transformed features (sanity check that the linear model agrees on the top features).
- Local: `top_k_contributions(explainer, row, k=5)` returns `(feature, value, shap_value)` for one flow; Phase 2 attaches it to each prediction so the agent receives "why the model fired" as evidence.
- A 1,000-row background sample is persisted as `explainer_background.parquet` so Phase 2 can rebuild the explainer without the training data.

## A13. MLflow experiment structure

Tracking URI: `sqlite:///$SECOPS_DATA_DIR/mlflow/mlflow.db` with artifacts beside it (overridable by `SECOPS_MLFLOW_TRACKING_URI` for a server later).

| Experiment | Runs |
|---|---|
| `secops/detection-binary` | logreg / xgboost / lightgbm × weighting {none, balanced}; split `chrono_within_group`; policy `relabel_benign` |
| `secops/detection-family` | xgboost / lightgbm × weighting; same split; attack rows only |
| `secops/detection-heldout` | champion binary config on `heldout_friday` with the champion's threshold reused |
| `secops/detection-ablations` | champion config with `v1-withport`; champion config with policy `drop` |

Every run: tags `dataset_variant`, `manifest_sha`, `git_sha`, `split_strategy`, `attempted_policy`, `feature_spec`, `model`, `weighting`, `task`; model params; metrics as in A10; artifacts as in A14. Registered models `secops-detector` and `secops-family-classifier`, alias `champion`; promotion rule: highest validation PR-AUC (binary) or validation macro-F1 (family), ties → simpler model.

Why MLflow (learning-mode note): we must compare ~15 runs on identical data, reproduce any of them, and hand a versioned model to an API by name rather than by file path. Alternatives: Weights & Biases (hosted, account needed), DVC (versions files, no registry/UI), spreadsheets (no artifacts). MLflow runs from a SQLite file locally and from a server later, and registry aliases are exactly the promotion mechanism Phases 2 and 6 need.

## A14. Expected artifacts

Under `$SECOPS_DATA_DIR`: `raw/CICIDS2017_improved.zip`, `raw/improved/*.csv`, `raw/MANIFEST_VERIFIED.json`; `processed/{relabel_benign,drop}/flows.parquet`; `reports/<policy>/cleaning_report.json` (rows in/out, inf cells, attempted rows, duplicates per label, label counts before/after) and `split_report.json` (counts per split × family, time ranges per split × day, leak check, Active/Idle sanity); `mlflow/`.

Per MLflow run: `metrics.json`, `threshold.json` (binary), `feature_spec.json`, `pr_curve.png` (binary), `confusion_matrix.png`, `per_label_recall.csv` (binary), `per_class_metrics.json` (family), `shap_global_importance.csv`, `shap_summary.png`, `explainer_background.parquet`, model (flavour-specific, with signature and input example).

In the repository: `docs/dataset.md`, `docs/evaluation.md` (ML section with measured tables), `docs/interview-notes.md` (Phase 1 entries), `data/README.md`, `configs/*.yaml`, README results section (measured numbers only), `tests/fixtures/mini_cicids/*.csv` (≤1,000-row sampled fixture, documented).

## A15. Tests

Unit (`tests/unit`, CI, no real data): schema constants; manifest hashing and verification; ingest header validation and dtypes; cleaning (inf, duplicates, attempted policies, family mapping, report counts); split (fractions, group coverage, leak check, held-out day); feature spec (order, missing columns, dtype, save/load); model factory (fit/predict on tiny data, sample weights, NaN tolerance, multiclass); metrics (hand-computed values); threshold selection (budget, tie rule, fallback); explainer (ranking, top-k ordering); plots (render and save); registry (promote + load round trip against a temporary tracking URI).

Integration (`tests/integration`, CI on the fixture): `build` on the fixture → Parquet with expected columns and clean reports; `run_training` on the fixture → MLflow run with expected metrics and artifacts for both tasks.

Full-data checks (local, not CI): assertions inside the build (row totals, no leak, every family in every chrono split) and the Phase 1 completion check in Task 13.

## A16. Phase 1 folder structure

```
agentic-security-operations/
├── README.md                         # updated: setup + Phase 1 results (measured only)
├── pyproject.toml  uv.lock  .python-version  Makefile  .gitignore  .env.example  .pre-commit-config.yaml
├── configs/
│   ├── data.yaml
│   └── models/
│       ├── logreg_binary.yaml  xgboost_binary.yaml  lightgbm_binary.yaml
│       ├── xgboost_family.yaml  lightgbm_family.yaml
│       ├── lightgbm_binary_heldout.yaml
│       └── ablation_withport.yaml  ablation_drop.yaml
├── data/README.md                    # source, terms, citations, download/verify/build steps
├── src/secops/
│   ├── __init__.py  config.py
│   ├── data/       __init__.py  schema.py  manifest.py  ingest.py  clean.py  split.py  build.py  cli.py
│   └── detection/  __init__.py  features.py  models.py  metrics.py  plots.py  explain.py  registry.py  train.py  cli.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/make_fixture.py  fixtures/mini_cicids/{monday..friday}.csv  fixtures/mini_cicids/README.md
│   ├── unit/test_config.py  unit/data/  unit/detection/
│   └── integration/
├── docs/  dataset.md  evaluation.md  interview-notes.md  superpowers/{specs,plans}/
└── .github/workflows/ci.yml          # ruff, mypy, unit + integration tests on the fixture
```

Folders from the Phase 0 layout that Phase 1 does not need (`schemas/`, `api/`, `agent/`, `tools/`, `db/`, `evaluation/`, `observability/`, `notebooks/`, `scripts/`) are not created yet.

## A17. Implementation milestones

| # | Milestone | Tasks | Verifiable output |
|---|---|---|---|
| M1 | Project scaffold | 1 | `uv run pytest` green; ruff/mypy clean; CI workflow runs |
| M2 | Schema, manifest, download/verify | 2, 3 | `secops-data verify` passes on the real files |
| M3 | Ingest + fixture | 4 | fixture CSVs committed; strict header validation tested |
| M4 | Clean + split + build | 5, 6 | `secops-data build` produces Parquet + reports on fixture and on full data |
| M5 | Metrics, threshold, features, models | 7, 8 | unit-tested library functions |
| M6 | Explainer, plots, training CLI | 9, 10 | six binary runs logged with all artifacts |
| M7 | Registry, family, held-out, ablations | 11, 12 | `secops-detector@champion` loadable; all experiments logged |
| M8 | Documentation + results | 13 | measured tables in docs and README; interview notes |

## A18. Risks and mitigation (Phase 1 specific)

| Risk | Mitigation |
|---|---|
| 8.7 GB RAM: 2.1 M rows with string columns | float32 features, `category` dtype for IPs/labels, `Flow ID` not loaded, per-day read then concat, Parquet via pyarrow; full build verified in Task 6 before any training |
| XGBoost/LightGBM wheels and OpenMP on WSL | pinned versions in `uv.lock`; `libgomp1` noted in README; CI runs the same versions |
| Validation used for early stopping and for threshold selection | documented optimism source; test is untouched |
| Dedup removes most port-scan rows, shifting class balance | counts logged before/after; caveat written next to every table |
| Fixture sampled from real data | ≤ 1,000 rows, cited in `tests/fixtures/mini_cicids/README.md`; a synthetic generator is the fallback if the terms are clarified against it |
| MLflow SQLite on DrvFs is slow / locks | tracking store under `$SECOPS_DATA_DIR` on the Linux filesystem |
| pyfunc probability contract differs across MLflow flavours | pinned by the registry test in Task 11; fails at implementation, not in Phase 2 |
| Over-claiming | all numbers in docs come from `secops-train report`, which reads MLflow |

## A19. Definition of "Phase 1 complete"

1. `make setup && uv run secops-data verify && make data-build && make train-all` reproduces every run from a clean clone, given the data directory.
2. `make test` green locally and in CI (ruff, mypy, unit, integration).
3. MLflow holds the binary (6 runs), family (4 runs), held-out (1 run) and ablation (2 runs) experiments with all artifacts in A14.
4. `secops-detector@champion` and `secops-family-classifier@champion` load via `registry.load_model` and reproduce the logged test metrics.
5. `docs/dataset.md`, `docs/evaluation.md`, `docs/interview-notes.md` and README contain measured tables; any unmeasured cell says `TBD`.
6. The build reports show zero leak-check failures and the documented dedup/attempted counts.

---

# Part B — Tasks

Each task is one commit on branch `phase-1/ml-detection` created from `main`. Part B is appended below.

### Task 1: Project scaffold (uv, package, tooling, CI)

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `src/secops/__init__.py`, `src/secops/config.py`, `tests/conftest.py`, `tests/unit/test_config.py`

**Interfaces:**
- Produces: `secops.config.Settings` with fields `data_dir: Path`, `mlflow_tracking_uri: str | None`, `random_seed: int`; properties `raw_dir`, `processed_dir`, `reports_dir`, `mlflow_dir`; method `resolved_tracking_uri() -> str`; function `get_settings() -> Settings` (cached).

- [ ] **Step 1: Install uv, pin Python 3.12, create the branch**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /mnt/d/Internship/Agentic-security-operations
git checkout -b phase-1/ml-detection
uv python install 3.12
echo "3.12" > .python-version
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "secops"
version = "0.1.0"
description = "Agentic Security Operations Platform: ML intrusion detection and evidence-grounded alert investigation"
requires-python = ">=3.12"
readme = "README.md"
dependencies = [
  "pandas>=2.2",
  "pyarrow>=16",
  "numpy>=1.26",
  "scikit-learn>=1.5",
  "xgboost>=2.1",
  "lightgbm>=4.4",
  "shap>=0.46",
  "mlflow>=2.16",
  "pydantic>=2.8",
  "pydantic-settings>=2.4",
  "pyyaml>=6",
  "typer>=0.12",
  "matplotlib>=3.9",
  "requests>=2.32",
  "tabulate>=0.9",
]

[project.scripts]
secops-data = "secops.data.cli:app"
secops-train = "secops.detection.cli:app"

[dependency-groups]
dev = [
  "pytest>=8",
  "pytest-cov>=5",
  "ruff>=0.6",
  "mypy>=1.11",
  "pandas-stubs>=2.2",
  "types-PyYAML",
  "types-requests",
  "pre-commit>=3.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/secops"]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "S", "N", "W"]
ignore = ["S101"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S", "B011"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
files = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["integration: slower tests that run the pipelines on the fixture"]
```

- [ ] **Step 3: Write `.gitignore`, `.env.example`, `Makefile`, pre-commit config, CI workflow**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.mypy_cache/
.ruff_cache/
.pytest_cache/
.coverage
htmlcov/
dist/
.env
mlruns/
mlartifacts/
data/raw/
data/processed/
*.parquet
*.zip
```

`.env.example`:
```
# Linux-native path; never under /mnt/d
SECOPS_DATA_DIR=/home/ramy/data/secops
# Leave empty to use sqlite under SECOPS_DATA_DIR/mlflow
SECOPS_MLFLOW_TRACKING_URI=
SECOPS_RANDOM_SEED=42
```

`Makefile`:
```makefile
.PHONY: setup lint type test-unit test-integration test data-download data-build train-all mlflow-ui

setup:
	uv sync --all-groups
	uv run pre-commit install

lint:
	uv run ruff check . && uv run ruff format --check .

type:
	uv run mypy

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration -m integration

test: lint type test-unit test-integration

data-download:
	uv run secops-data download

data-build:
	uv run secops-data build --attempted-policy relabel_benign
	uv run secops-data build --attempted-policy drop

train-all:
	for c in configs/models/*.yaml; do uv run secops-train run --config $$c; done

mlflow-ui:
	uv run mlflow ui --backend-store-uri "$$(uv run python -c 'from secops.config import get_settings as g; print(g().resolved_tracking_uri())')"
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks
```

`.github/workflows/ci.yml`:
```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]
jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv sync --all-groups
      - run: uv run ruff check . && uv run ruff format --check .
      - run: uv run mypy
      - run: uv run pytest tests/unit
      - run: uv run pytest tests/integration -m integration
```

- [ ] **Step 4: Write the failing config test**

`tests/unit/test_config.py`:
```python
from pathlib import Path

from secops.config import Settings


def test_settings_defaults_derive_subdirectories(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path)
    assert s.raw_dir == tmp_path / "raw"
    assert s.processed_dir == tmp_path / "processed"
    assert s.reports_dir == tmp_path / "reports"
    assert s.mlflow_dir == tmp_path / "mlflow"
    assert s.random_seed == 42


def test_tracking_uri_defaults_to_sqlite_under_data_dir(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, mlflow_tracking_uri=None)
    assert s.resolved_tracking_uri() == f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}"


def test_tracking_uri_override_wins(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, mlflow_tracking_uri="http://localhost:5000")
    assert s.resolved_tracking_uri() == "http://localhost:5000"
```

- [ ] **Step 5: Run to verify it fails**

Run: `uv sync --all-groups && uv run pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.config'`

- [ ] **Step 6: Implement `src/secops/__init__.py`, `src/secops/config.py`, `tests/conftest.py`**

`src/secops/__init__.py`:
```python
"""Agentic Security Operations Platform."""

__version__ = "0.1.0"
```

`src/secops/config.py`:
```python
"""Runtime settings. Paths and secrets come from the environment, never from code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECOPS_", env_file=".env", extra="ignore")

    data_dir: Path = Path.home() / "data" / "secops"
    mlflow_tracking_uri: str | None = None
    random_seed: int = 42

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def mlflow_dir(self) -> Path:
        return self.data_dir / "mlflow"

    def resolved_tracking_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return f"sqlite:///{self.mlflow_dir / 'mlflow.db'}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

`tests/conftest.py`:
```python
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mini_cicids"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR
```

- [ ] **Step 7: Run tests, lint, type-check**

Run: `uv run pytest tests/unit -v && uv run ruff check . && uv run ruff format . && uv run mypy`
Expected: 3 passed; ruff and mypy report no errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore .env.example Makefile .pre-commit-config.yaml .github/workflows/ci.yml src/secops tests/conftest.py tests/unit/test_config.py
git commit -m "chore: scaffold secops package with uv, ruff, mypy, pytest and CI

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Dataset schema constants and family mapping

**Files:**
- Create: `src/secops/data/__init__.py`, `src/secops/data/schema.py`, `tests/unit/data/__init__.py`, `tests/unit/data/test_schema.py`

**Interfaces:**
- Produces: `RAW_COLUMNS: list[str]` (91 names in file order), `ID_COLS`, `META_COLS`, `LABEL_COL = "Label"`, `ATTEMPTED_COL = "Attempted Category"`, `FEATURE_COLS: list[str]` (82), `PORT_COL = "Dst Port"`, `BENIGN_LABEL = "BENIGN"`, `ATTEMPTED_SUFFIX = " - Attempted"`, `FAMILY_MAP: dict[str, str]`, `FAMILIES: list[str]`, `FAMILY_CLASSIFIER_CLASSES: list[str]` (families minus `rare_exploit`), `DAY_FILES: dict[str, str]`, `DAY_ORDER: list[str]`; functions `base_label(label) -> str`, `is_attempted_label(label) -> bool`, `family_of(label) -> str` (returns `"benign"` for BENIGN, raises `KeyError` for unknown labels).

- [ ] **Step 1: Write the failing tests**

`tests/unit/data/test_schema.py`:
```python
import pytest

from secops.data import schema as s


def test_raw_columns_count_and_boundaries() -> None:
    assert len(s.RAW_COLUMNS) == 91
    assert s.RAW_COLUMNS[0] == "id"
    assert s.RAW_COLUMNS[-2:] == ["Label", "Attempted Category"]


def test_feature_columns_exclude_identifiers_and_labels() -> None:
    assert len(s.FEATURE_COLS) == 82
    for c in s.ID_COLS + s.META_COLS + [s.LABEL_COL, s.ATTEMPTED_COL]:
        assert c not in s.FEATURE_COLS
    assert "Protocol" in s.FEATURE_COLS
    assert "Total TCP Flow Time" in s.FEATURE_COLS
    assert s.PORT_COL not in s.FEATURE_COLS


def test_base_label_strips_attempted_suffix() -> None:
    assert s.base_label("DoS Hulk - Attempted") == "DoS Hulk"
    assert s.base_label("DoS Hulk") == "DoS Hulk"
    assert s.is_attempted_label("Botnet - Attempted")
    assert not s.is_attempted_label("Botnet")


def test_family_of_every_known_label() -> None:
    assert s.family_of("BENIGN") == "benign"
    assert s.family_of("FTP-Patator") == "brute_force"
    assert s.family_of("Infiltration - Portscan") == "port_scan"
    assert s.family_of("Heartbleed") == "rare_exploit"
    assert s.family_of("Web Attack - XSS - Attempted") == "web_attack"


def test_family_of_unknown_label_raises() -> None:
    with pytest.raises(KeyError):
        s.family_of("Not A Label")


def test_family_classifier_classes_exclude_rare() -> None:
    assert "rare_exploit" not in s.FAMILY_CLASSIFIER_CLASSES
    assert set(s.FAMILY_CLASSIFIER_CLASSES) == {
        "brute_force",
        "dos",
        "ddos",
        "port_scan",
        "web_attack",
        "botnet",
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/data/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.data'`

- [ ] **Step 3: Implement `schema.py`** (`src/secops/data/__init__.py` is a one-line docstring)

```python
"""Column layout, label vocabulary and family grouping for DistriNet-improved CIC-IDS-2017.

Measured on the 2023-04-27 release of CICIDS2017_improved.zip (see data/README.md).
"""

from __future__ import annotations

DAY_ORDER: list[str] = ["monday", "tuesday", "wednesday", "thursday", "friday"]
DAY_FILES: dict[str, str] = {d: f"{d}.csv" for d in DAY_ORDER}

ID_COLS: list[str] = ["id", "Flow ID"]
META_COLS: list[str] = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]
LABEL_COL = "Label"
ATTEMPTED_COL = "Attempted Category"
PORT_COL = "Dst Port"
BENIGN_LABEL = "BENIGN"
ATTEMPTED_SUFFIX = " - Attempted"

RAW_COLUMNS: list[str] = [
    "id",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd RST Flags",
    "Bwd RST Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Fwd Segment Size Avg",
    "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "ICMP Code",
    "ICMP Type",
    "Total TCP Flow Time",
    "Label",
    "Attempted Category",
]

_NON_FEATURE = set(ID_COLS) | set(META_COLS) | {LABEL_COL, ATTEMPTED_COL}
FEATURE_COLS: list[str] = [c for c in RAW_COLUMNS if c not in _NON_FEATURE]

FAMILY_MAP: dict[str, str] = {
    "FTP-Patator": "brute_force",
    "SSH-Patator": "brute_force",
    "DoS Hulk": "dos",
    "DoS GoldenEye": "dos",
    "DoS Slowloris": "dos",
    "DoS Slowhttptest": "dos",
    "DDoS": "ddos",
    "Portscan": "port_scan",
    "Infiltration - Portscan": "port_scan",
    "Web Attack - Brute Force": "web_attack",
    "Web Attack - XSS": "web_attack",
    "Web Attack - SQL Injection": "web_attack",
    "Botnet": "botnet",
    "Heartbleed": "rare_exploit",
    "Infiltration": "rare_exploit",
}
FAMILIES: list[str] = sorted(set(FAMILY_MAP.values()))
FAMILY_CLASSIFIER_CLASSES: list[str] = [f for f in FAMILIES if f != "rare_exploit"]


def is_attempted_label(label: str) -> bool:
    return label.endswith(ATTEMPTED_SUFFIX)


def base_label(label: str) -> str:
    return label[: -len(ATTEMPTED_SUFFIX)] if is_attempted_label(label) else label


def family_of(label: str) -> str:
    """Family for a raw label. Attempted labels map to their base family. Raises KeyError."""
    base = base_label(label)
    if base == BENIGN_LABEL:
        return "benign"
    return FAMILY_MAP[base]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/data/test_schema.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/secops/data tests/unit/data
git commit -m "feat(data): column schema, label vocabulary and attack-family mapping

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Data manifest, download and verification CLI

**Files:**
- Create: `src/secops/data/manifest.py`, `src/secops/data/cli.py`, `data/README.md`, `tests/unit/data/test_manifest.py`

**Interfaces:**
- Produces: `FileEntry(name, size_bytes, rows, sha256)` frozen dataclass; `ZIP_URL`, `ZIP_SHA256`, `ZIP_SIZE`, `EXTRACT_SUBDIR = "improved"`, `MANIFEST: dict[str, FileEntry]` keyed by day; `sha256_of(path) -> str`; `manifest_digest() -> str`; `csv_path(raw_dir, day) -> Path`; `verify_raw_dir(raw_dir) -> list[str]` (problems; empty means OK); `download_zip(url, dest, expected_sha256) -> Path`; `extract_zip(zip_path, dest_dir) -> list[Path]`; `write_verification_stamp(raw_dir) -> Path`.
- CLI: `secops-data download [--data-dir PATH]`, `secops-data verify [--data-dir PATH]`.

- [ ] **Step 1: Write failing tests**

`tests/unit/data/test_manifest.py`:
```python
import hashlib
from pathlib import Path

from secops.data import manifest as m


def test_manifest_lists_five_days_with_hashes() -> None:
    assert set(m.MANIFEST) == {"monday", "tuesday", "wednesday", "thursday", "friday"}
    for entry in m.MANIFEST.values():
        assert len(entry.sha256) == 64
        assert entry.size_bytes > 0 and entry.rows > 0
    assert sum(e.rows for e in m.MANIFEST.values()) == 2_099_976


def test_sha256_of_small_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert m.sha256_of(p) == hashlib.sha256(b"hello").hexdigest()


def test_verify_raw_dir_reports_missing_and_mismatched(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "improved"
    raw.mkdir(parents=True)
    (raw / "monday.csv").write_text("not the real file")
    problems = m.verify_raw_dir(tmp_path / "raw")
    assert any("monday.csv" in p and "sha256" in p for p in problems)
    assert any("tuesday.csv" in p and "missing" in p for p in problems)


def test_manifest_digest_is_stable() -> None:
    assert m.manifest_digest() == m.manifest_digest()
    assert len(m.manifest_digest()) == 64
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/data/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.data.manifest'`

- [ ] **Step 3: Implement `manifest.py`**

```python
"""Expected raw files for the DistriNet-improved CIC-IDS-2017 release, with verification helpers.

Hashes measured 2026-10-03 from CICIDS2017_improved.zip (server mtime 2023-04-27).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from secops.data.schema import DAY_FILES

ZIP_URL = (
    "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"
)
ZIP_SHA256 = "97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883"
ZIP_SIZE = 343_549_013
EXTRACT_SUBDIR = "improved"


@dataclass(frozen=True)
class FileEntry:
    name: str
    size_bytes: int
    rows: int
    sha256: str


MANIFEST: dict[str, FileEntry] = {
    "monday": FileEntry(
        "monday.csv",
        207_875_155,
        371_624,
        "51fe5dc962626efb4ae70dce0303072fb780da0932822b651202ee9c2fbc1aff",
    ),
    "tuesday": FileEntry(
        "tuesday.csv",
        178_397_720,
        322_078,
        "e2a0a5b631dfc6b455cc9f9a88b944110637d70a7f74171473925f76f38b6b0c",
    ),
    "wednesday": FileEntry(
        "wednesday.csv",
        291_290_505,
        496_641,
        "bf46c5f3c792e8817381f724511229569606918eaf07ac986d7a2592b6341bc2",
    ),
    "thursday": FileEntry(
        "thursday.csv",
        189_519_159,
        362_076,
        "78a4d11eaf473d099e30e71ddb01e0f38218e844c0a9cdd36602145d674af482",
    ),
    "friday": FileEntry(
        "friday.csv",
        285_188_226,
        547_557,
        "ebd499e6f23bd59f9cb81bec28178491b02b925fa5640a24215c9437d79482d0",
    ),
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_digest() -> str:
    joined = "|".join(f"{d}:{MANIFEST[d].sha256}" for d in sorted(MANIFEST))
    return hashlib.sha256(joined.encode()).hexdigest()


def csv_path(raw_dir: Path, day: str) -> Path:
    return raw_dir / EXTRACT_SUBDIR / DAY_FILES[day]


def verify_raw_dir(raw_dir: Path) -> list[str]:
    problems: list[str] = []
    for day, entry in MANIFEST.items():
        p = csv_path(raw_dir, day)
        if not p.exists():
            problems.append(f"{entry.name}: missing at {p}")
            continue
        size = p.stat().st_size
        if size != entry.size_bytes:
            problems.append(f"{entry.name}: size {size} != expected {entry.size_bytes}")
        digest = sha256_of(p)
        if digest != entry.sha256:
            problems.append(f"{entry.name}: sha256 {digest} != expected {entry.sha256}")
    return problems


def download_zip(url: str, dest: Path, expected_sha256: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest) == expected_sha256:
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    digest = sha256_of(dest)
    if digest != expected_sha256:
        raise ValueError(f"downloaded zip sha256 {digest} != expected {expected_sha256}")
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        z.extractall(dest_dir, members=names)
    return [dest_dir / n for n in names]


def write_verification_stamp(raw_dir: Path) -> Path:
    stamp = raw_dir / "MANIFEST_VERIFIED.json"
    payload = {
        "verified_at": datetime.now(UTC).isoformat(),
        "manifest_digest": manifest_digest(),
        "files": {d: asdict(e) for d, e in MANIFEST.items()},
    }
    stamp.write_text(json.dumps(payload, indent=2))
    return stamp
```

- [ ] **Step 4: Implement `cli.py` (`build` is added in Task 6)**

```python
"""secops-data command line."""

from __future__ import annotations

from pathlib import Path

import typer

from secops.config import get_settings
from secops.data import manifest as m

app = typer.Typer(help="Dataset acquisition and preprocessing.")


def _report_problems(problems: list[str]) -> None:
    for p in problems:
        typer.echo(f"PROBLEM: {p}", err=True)
    if problems:
        raise typer.Exit(code=1)


@app.command()
def download(data_dir: Path | None = None) -> None:
    """Download the DistriNet-improved CIC-IDS-2017 zip, verify, extract, verify CSVs."""
    raw = (data_dir or get_settings().data_dir) / "raw"
    zip_path = m.download_zip(m.ZIP_URL, raw / "CICIDS2017_improved.zip", m.ZIP_SHA256)
    typer.echo(f"zip verified: {zip_path}")
    m.extract_zip(zip_path, raw / m.EXTRACT_SUBDIR)
    _report_problems(m.verify_raw_dir(raw))
    stamp = m.write_verification_stamp(raw)
    typer.echo(f"all files verified; stamp written to {stamp}")


@app.command()
def verify(data_dir: Path | None = None) -> None:
    """Verify already-downloaded CSVs against the manifest."""
    raw = (data_dir or get_settings().data_dir) / "raw"
    _report_problems(m.verify_raw_dir(raw))
    typer.echo("all files verified")
```

- [ ] **Step 5: Write `data/README.md`**

Contents: dataset name; the three-layer table from A1; source URLs; the measured hash table from A2; the terms-of-use statement (research use, cite both papers, no redistribution, nothing committed); the two BibTeX entries below; the commands `make data-download` and `make data-build`.

```bibtex
@inproceedings{sharafaldin2018toward,
  title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A.},
  booktitle={Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)},
  year={2018}
}
@inproceedings{liu2022error,
  title={Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018},
  author={Liu, Lisa and Engelen, Gints and Lynar, Timothy and Essam, Daryl and Joosen, Wouter},
  booktitle={2022 IEEE Conference on Communications and Network Security (CNS)},
  pages={254--262},
  year={2022},
  organization={IEEE}
}
```

- [ ] **Step 6: Run tests, then verify against the real files already present**

Run: `uv run pytest tests/unit/data/test_manifest.py -v`
Expected: 4 passed

Run: `SECOPS_DATA_DIR=$HOME/data/secops uv run secops-data verify`
Expected: `all files verified`

- [ ] **Step 7: Commit**

```bash
git add src/secops/data/manifest.py src/secops/data/cli.py data/README.md tests/unit/data/test_manifest.py
git commit -m "feat(data): manifest with measured hashes, download/verify CLI, dataset README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Ingest with strict header validation, plus the test fixture

**Files:**
- Create: `src/secops/data/ingest.py`, `tests/fixtures/make_fixture.py`, `tests/fixtures/mini_cicids/{monday,tuesday,wednesday,thursday,friday}.csv`, `tests/fixtures/mini_cicids/README.md`, `tests/unit/data/test_ingest.py`

**Interfaces:**
- Produces: `class SchemaError(ValueError)`; `OUTPUT_COLUMNS = ["id", "day", *META_COLS, *FEATURE_COLS, LABEL_COL, ATTEMPTED_COL]`; `read_day(path: Path, day: str) -> pd.DataFrame` (features float32, `Timestamp` tz-aware UTC, `Src IP`/`Dst IP`/`Label`/`day` categorical; raises `SchemaError` naming missing and unexpected columns; raises `ValueError` for an unknown day); `read_all(raw_dir: Path, days: list[str] | None = None, subdir: str = "improved") -> pd.DataFrame` (concatenated in `DAY_ORDER`, `day` ordered categorical).

- [ ] **Step 1: Create the fixture by stratified sampling of the real CSVs**

`tests/fixtures/make_fixture.py` (run once; committed for reproducibility):
```python
"""Build a <=1,000-row fixture from the real improved CSVs: every label present, time order kept.

Usage: SECOPS_DATA_DIR=~/data/secops uv run python tests/fixtures/make_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from secops.config import get_settings
from secops.data.manifest import csv_path
from secops.data.schema import DAY_ORDER, LABEL_COL

OUT = Path(__file__).parent / "mini_cicids"
PER_LABEL_CAP = 40
BENIGN_PER_DAY = 120
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = get_settings().raw_dir
    for day in DAY_ORDER:
        df = pd.read_csv(csv_path(raw, day), low_memory=False)
        parts = []
        for label, g in df.groupby(LABEL_COL, sort=False):
            cap = BENIGN_PER_DAY if label == "BENIGN" else PER_LABEL_CAP
            take = g if len(g) <= cap else g.iloc[np.sort(rng.choice(len(g), cap, replace=False))]
            parts.append(take)
        out = pd.concat(parts).sort_values("id")
        out.to_csv(OUT / f"{day}.csv", index=False)
        print(day, len(out))


if __name__ == "__main__":
    main()
```

Run it once. Then write `tests/fixtures/mini_cicids/README.md`: what the fixture is (a ≤1,000-row research-use sample of the DistriNet-improved CIC-IDS-2017, the two citations from Task 3), and the exact regeneration command. Expected fixture size: 5 × 120 benign + capped attack labels ≈ 900 rows.

- [ ] **Step 2: Write failing tests**

`tests/unit/data/test_ingest.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secops.data import schema as s
from secops.data.ingest import SchemaError, read_all, read_day


def test_read_day_returns_expected_columns_and_dtypes(fixture_dir: Path) -> None:
    df = read_day(fixture_dir / "tuesday.csv", "tuesday")
    expected = ["id", "day", *s.META_COLS, *s.FEATURE_COLS, s.LABEL_COL, s.ATTEMPTED_COL]
    assert list(df.columns) == expected
    assert all(df[c].dtype == np.float32 for c in s.FEATURE_COLS)
    assert str(df["Timestamp"].dtype) == "datetime64[ns, UTC]"
    assert df["day"].iloc[0] == "tuesday"
    assert (df[s.LABEL_COL] == "FTP-Patator").any()


def test_read_day_rejects_wrong_header(tmp_path: Path) -> None:
    bad = tmp_path / "monday.csv"
    bad.write_text("id,Flow ID,Src IP,Label\n1,x,1.1.1.1,BENIGN\n")
    with pytest.raises(SchemaError) as exc:
        read_day(bad, "monday")
    assert "missing" in str(exc.value) and "Flow Duration" in str(exc.value)


def test_read_day_rejects_unknown_day(fixture_dir: Path) -> None:
    with pytest.raises(ValueError):
        read_day(fixture_dir / "monday.csv", "sunday")


def test_read_all_concatenates_days_in_order(fixture_dir: Path) -> None:
    df = read_all(fixture_dir, days=["monday", "tuesday"], subdir="")
    assert list(df["day"].cat.categories) == ["monday", "tuesday"]
    assert df["day"].cat.codes.is_monotonic_increasing
    assert (df["day"] == "tuesday").sum() > 0


def test_timestamps_are_within_known_capture_window(fixture_dir: Path) -> None:
    df = read_day(fixture_dir / "friday.csv", "friday")
    assert df["Timestamp"].min() >= pd.Timestamp("2017-07-07", tz="UTC")
    assert df["Timestamp"].max() < pd.Timestamp("2017-07-08", tz="UTC")
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/data/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.data.ingest'`

- [ ] **Step 4: Implement `ingest.py`**

```python
"""Read raw day CSVs into typed frames. The header must match the improved-dataset layout exactly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from secops.data import schema as s


class SchemaError(ValueError):
    """Raised when a CSV header does not match RAW_COLUMNS."""


OUTPUT_COLUMNS: list[str] = [
    "id",
    "day",
    *s.META_COLS,
    *s.FEATURE_COLS,
    s.LABEL_COL,
    s.ATTEMPTED_COL,
]


def _validate_header(path: Path) -> None:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in s.RAW_COLUMNS if c not in header]
    unexpected = [c for c in header if c not in s.RAW_COLUMNS]
    if missing or unexpected or header != s.RAW_COLUMNS:
        raise SchemaError(
            f"{path.name}: header mismatch. missing={missing} unexpected={unexpected} "
            f"ordered_match={header == s.RAW_COLUMNS}"
        )


def read_day(path: Path, day: str) -> pd.DataFrame:
    if day not in s.DAY_ORDER:
        raise ValueError(f"unknown day {day!r}; expected one of {s.DAY_ORDER}")
    _validate_header(path)
    dtypes: dict[str, str] = {c: "float32" for c in s.FEATURE_COLS}
    dtypes.update(
        {
            "id": "int64",
            "Src IP": "category",
            "Dst IP": "category",
            "Src Port": "int32",
            "Dst Port": "int32",
            s.LABEL_COL: "category",
            s.ATTEMPTED_COL: "int8",
        }
    )
    usecols = [c for c in s.RAW_COLUMNS if c != "Flow ID"]
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes, low_memory=False)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["day"] = pd.Categorical([day] * len(df), categories=s.DAY_ORDER)
    for c in s.FEATURE_COLS:
        if df[c].dtype != np.float32:
            df[c] = df[c].astype(np.float32)
    return df[OUTPUT_COLUMNS]


def read_all(
    raw_dir: Path, days: list[str] | None = None, subdir: str = "improved"
) -> pd.DataFrame:
    days = days or s.DAY_ORDER
    frames = [read_day(raw_dir / subdir / s.DAY_FILES[d], d) for d in days]
    out = pd.concat(frames, ignore_index=True)
    out["day"] = pd.Categorical(out["day"].astype(str), categories=days, ordered=True)
    out[s.LABEL_COL] = out[s.LABEL_COL].astype(str).astype("category")
    return out
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/data/test_ingest.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/secops/data/ingest.py tests/fixtures tests/unit/data/test_ingest.py
git commit -m "feat(data): strict-header ingest and sampled test fixture

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Cleaning (infinity, Attempted policy, duplicates, family) with a report

**Files:**
- Create: `src/secops/data/clean.py`, `tests/unit/data/test_clean.py`

**Interfaces:**
- Produces: `class AttemptedPolicy(StrEnum): RELABEL_BENIGN = "relabel_benign"; DROP = "drop"`; `@dataclass CleaningReport(rows_in, rows_out, inf_cells_replaced, attempted_rows, attempted_policy, duplicates_removed, duplicates_removed_by_label, label_counts_in, label_counts_out, family_counts_out)` with `to_json(path) -> Path`; `clean(df, policy) -> tuple[pd.DataFrame, CleaningReport]`. Output adds `label_raw` (original), `label` (post-policy), `family`, `is_attack` (int8); keeps everything from ingest; index reset.

- [ ] **Step 1: Write failing tests**

`tests/unit/data/test_clean.py`:
```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from secops.data import schema as s
from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all


def _df(fixture_dir: Path) -> pd.DataFrame:
    return read_all(fixture_dir, subdir="")


def test_clean_replaces_inf_with_nan(fixture_dir: Path) -> None:
    df = _df(fixture_dir)
    df.loc[df.index[0], "Flow Bytes/s"] = np.float32("inf")
    df.loc[df.index[1], "Flow Packets/s"] = np.float32("-inf")
    out, rep = clean(df, AttemptedPolicy.RELABEL_BENIGN)
    assert rep.inf_cells_replaced == 2
    assert not np.isinf(out[s.FEATURE_COLS].to_numpy()).any()


def test_relabel_benign_policy_turns_attempted_into_benign(fixture_dir: Path) -> None:
    df = _df(fixture_dir)
    n_attempted = df[s.LABEL_COL].astype(str).str.endswith(s.ATTEMPTED_SUFFIX).sum()
    out, rep = clean(df, AttemptedPolicy.RELABEL_BENIGN)
    assert rep.attempted_rows == n_attempted
    assert not out["label"].astype(str).str.endswith(s.ATTEMPTED_SUFFIX).any()
    attempted = out[out["label_raw"].astype(str).str.endswith(s.ATTEMPTED_SUFFIX)]
    assert (attempted["label"] == s.BENIGN_LABEL).all()
    assert (attempted["is_attack"] == 0).all()


def test_drop_policy_removes_attempted_rows(fixture_dir: Path) -> None:
    out, rep = clean(_df(fixture_dir), AttemptedPolicy.DROP)
    assert not out["label_raw"].astype(str).str.endswith(s.ATTEMPTED_SUFFIX).any()
    assert rep.rows_out == rep.rows_in - rep.attempted_rows - rep.duplicates_removed


def test_exact_duplicates_are_removed_and_counted_by_label(fixture_dir: Path) -> None:
    df = _df(fixture_dir)
    dup = df.iloc[[5, 5, 5]].copy()
    dup["id"] = [10_000_001, 10_000_002, 10_000_003]
    df2 = pd.concat([df, dup], ignore_index=True)
    _, rep_base = clean(df, AttemptedPolicy.RELABEL_BENIGN)
    _, rep = clean(df2, AttemptedPolicy.RELABEL_BENIGN)
    assert rep.duplicates_removed == rep_base.duplicates_removed + 3
    assert sum(rep.duplicates_removed_by_label.values()) == rep.duplicates_removed


def test_family_and_is_attack_are_consistent(fixture_dir: Path) -> None:
    out, rep = clean(_df(fixture_dir), AttemptedPolicy.RELABEL_BENIGN)
    assert set(out.loc[out["is_attack"] == 0, "family"].astype(str).unique()) == {"benign"}
    assert "benign" not in set(out.loc[out["is_attack"] == 1, "family"].astype(str).unique())
    assert out.loc[out["label"] == "Heartbleed", "family"].astype(str).eq("rare_exploit").all()
    assert sum(rep.family_counts_out.values()) == rep.rows_out


def test_report_round_trips_to_json(fixture_dir: Path, tmp_path: Path) -> None:
    _, rep = clean(_df(fixture_dir), AttemptedPolicy.RELABEL_BENIGN)
    p = rep.to_json(tmp_path / "r.json")
    loaded = json.loads(p.read_text())
    assert loaded["attempted_policy"] == "relabel_benign"
    assert loaded["rows_out"] == rep.rows_out
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/data/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.data.clean'`

- [ ] **Step 3: Implement `clean.py`**

```python
"""Cleaning: infinity handling, Attempted-flow policy, exact-duplicate removal, family labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from secops.data import schema as s


class AttemptedPolicy(StrEnum):
    RELABEL_BENIGN = "relabel_benign"  # dataset authors' recommendation (default)
    DROP = "drop"  # ablation


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    inf_cells_replaced: int
    attempted_rows: int
    attempted_policy: str
    duplicates_removed: int
    duplicates_removed_by_label: dict[str, int] = field(default_factory=dict)
    label_counts_in: dict[str, int] = field(default_factory=dict)
    label_counts_out: dict[str, int] = field(default_factory=dict)
    family_counts_out: dict[str, int] = field(default_factory=dict)

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def clean(df: pd.DataFrame, policy: AttemptedPolicy) -> tuple[pd.DataFrame, CleaningReport]:
    rows_in = len(df)
    out = df.copy()
    out["label_raw"] = out[s.LABEL_COL].astype(str)
    label_counts_in = _counts(out["label_raw"])

    attempted_mask = out["label_raw"].str.endswith(s.ATTEMPTED_SUFFIX)
    attempted_rows = int(attempted_mask.sum())
    if policy is AttemptedPolicy.DROP:
        out = out.loc[~attempted_mask].copy()
        out["label"] = out["label_raw"]
    else:
        out["label"] = np.where(attempted_mask, s.BENIGN_LABEL, out["label_raw"])

    feats = out[s.FEATURE_COLS].to_numpy(dtype=np.float32)
    inf_mask = np.isinf(feats)
    inf_cells = int(inf_mask.sum())
    if inf_cells:
        out[s.FEATURE_COLS] = np.where(inf_mask, np.nan, feats).astype(np.float32)

    dup_mask = out.duplicated(subset=[*s.FEATURE_COLS, "label"], keep="first")
    dup_by_label = _counts(out.loc[dup_mask, "label"])
    out = out.loc[~dup_mask].copy()

    out["family"] = out["label"].map(s.family_of).astype("category")
    out["is_attack"] = (out["label"] != s.BENIGN_LABEL).astype(np.int8)
    out["label"] = out["label"].astype("category")
    out["label_raw"] = out["label_raw"].astype("category")

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(out),
        inf_cells_replaced=inf_cells,
        attempted_rows=attempted_rows,
        attempted_policy=str(policy),
        duplicates_removed=int(dup_mask.sum()),
        duplicates_removed_by_label=dup_by_label,
        label_counts_in=label_counts_in,
        label_counts_out=_counts(out["label"]),
        family_counts_out=_counts(out["family"]),
    )
    return out.reset_index(drop=True), report
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/data/test_clean.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/secops/data/clean.py tests/unit/data/test_clean.py
git commit -m "feat(data): cleaning with attempted-flow policy, dedup and family labels

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Chronological splits, leak check, and the `build` command

**Files:**
- Create: `src/secops/data/split.py`, `src/secops/data/build.py`, `configs/data.yaml`, `tests/unit/data/test_split.py`, `tests/integration/__init__.py`, `tests/integration/test_build_pipeline.py`
- Modify: `src/secops/data/cli.py` (add `build`)

**Interfaces:**
- Produces: `class SplitStrategy(StrEnum): CHRONO_WITHIN_GROUP = "chrono_within_group"; HELDOUT_FRIDAY = "heldout_friday"`; `SPLIT_COLUMN: dict[SplitStrategy, str]` = `{CHRONO_WITHIN_GROUP: "split_chrono", HELDOUT_FRIDAY: "split_heldout"}`; `class LeakError(ValueError)`; `assign_chrono_within_group(df, train_frac=0.70, val_frac=0.15, group_cols=("day","label")) -> pd.Series` of `"train"|"val"|"test"`; `assign_heldout_day(df, test_day="friday", val_frac=0.15) -> pd.Series`; `check_no_time_leak(df, split, group_cols=("day","label")) -> None`; `split_report(df, split, name) -> dict`; `build(raw_dir, processed_dir, reports_dir, policy, subdir="improved", train_frac=0.70, val_frac=0.15) -> Path`.

- [ ] **Step 1: Write failing unit tests**

`tests/unit/data/test_split.py`:
```python
from pathlib import Path

import pandas as pd
import pytest

from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all
from secops.data.split import (
    LeakError,
    assign_chrono_within_group,
    assign_heldout_day,
    check_no_time_leak,
    split_report,
)


@pytest.fixture
def cleaned(fixture_dir: Path) -> pd.DataFrame:
    df, _ = clean(read_all(fixture_dir, subdir=""), AttemptedPolicy.RELABEL_BENIGN)
    return df


def test_chrono_split_fractions_and_coverage(cleaned: pd.DataFrame) -> None:
    split = assign_chrono_within_group(cleaned)
    assert set(split.unique()) <= {"train", "val", "test"}
    frac = split.value_counts(normalize=True)
    assert 0.55 < frac["train"] < 0.85
    d = cleaned.assign(split=split)
    for _, grp in d.groupby(["day", "label"], observed=True):
        if len(grp) >= 7:
            assert set(grp["split"]) == {"train", "val", "test"}


def test_chrono_split_is_time_ordered_within_group(cleaned: pd.DataFrame) -> None:
    split = assign_chrono_within_group(cleaned)
    check_no_time_leak(cleaned, split)  # must not raise
    d = cleaned.assign(split=split)
    for _, grp in d.groupby(["day", "label"], observed=True):
        if {"train", "test"} <= set(grp["split"]):
            tr = grp.loc[grp.split == "train", "Timestamp"].max()
            te = grp.loc[grp.split == "test", "Timestamp"].min()
            assert tr <= te


def test_check_no_time_leak_raises(cleaned: pd.DataFrame) -> None:
    split = assign_chrono_within_group(cleaned).copy()
    d = cleaned.assign(split=split)
    earliest = d[(d.day == "monday") & (d.label == "BENIGN")].sort_values("Timestamp").index[0]
    split.loc[earliest] = "test"
    with pytest.raises(LeakError):
        check_no_time_leak(cleaned, split)


def test_heldout_day_puts_all_friday_in_test(cleaned: pd.DataFrame) -> None:
    split = assign_heldout_day(cleaned, test_day="friday")
    assert (split[cleaned.day == "friday"] == "test").all()
    assert (split[cleaned.day != "friday"] != "test").all()
    assert set(split[cleaned.day != "friday"].unique()) == {"train", "val"}


def test_invalid_fractions_rejected(cleaned: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        assign_chrono_within_group(cleaned, train_frac=0.9, val_frac=0.2)


def test_split_report_has_counts_and_ranges(cleaned: pd.DataFrame) -> None:
    split = assign_chrono_within_group(cleaned)
    rep = split_report(cleaned, split, "chrono_within_group")
    assert rep["name"] == "chrono_within_group"
    assert rep["rows"] == len(cleaned)
    assert set(rep["counts_by_split_family"]) == {"train", "val", "test"}
    assert rep["leak_check"] == "ok"
    assert rep["active_idle_max_seconds"] < 86_400
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/data/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.data.split'`

- [ ] **Step 3: Implement `split.py`**

```python
"""Chronological split assignment and leak checks."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from secops.data import schema as s


class SplitStrategy(StrEnum):
    CHRONO_WITHIN_GROUP = "chrono_within_group"
    HELDOUT_FRIDAY = "heldout_friday"


SPLIT_COLUMN: dict[SplitStrategy, str] = {
    SplitStrategy.CHRONO_WITHIN_GROUP: "split_chrono",
    SplitStrategy.HELDOUT_FRIDAY: "split_heldout",
}


class LeakError(ValueError):
    """A val/test row precedes a train row within the same group."""


def _time_quantile_within_group(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    """Position of each row inside its group after sorting by time: (rank+1)/size in (0, 1]."""
    ordered = df.sort_values(["Timestamp", "id"])
    g = ordered.groupby(list(group_cols), observed=True)
    rank = g.cumcount()
    size = g["id"].transform("size")
    return ((rank + 1) / size).reindex(df.index)


def assign_chrono_within_group(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    group_cols: Sequence[str] = ("day", "label"),
) -> pd.Series:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("fractions must be in (0,1) and sum to < 1")
    q = _time_quantile_within_group(df, group_cols)
    out = np.where(q <= train_frac, "train", np.where(q <= train_frac + val_frac, "val", "test"))
    return pd.Series(out, index=df.index, name="split")


def assign_heldout_day(
    df: pd.DataFrame, test_day: str = "friday", val_frac: float = 0.15
) -> pd.Series:
    if test_day not in s.DAY_ORDER:
        raise ValueError(f"unknown day {test_day}")
    is_test = (df["day"].astype(str) == test_day).to_numpy()
    rest = df.loc[~is_test]
    q = _time_quantile_within_group(rest, ("day", "label"))
    out = pd.Series("test", index=df.index, name="split")
    out.loc[rest.index] = np.where(q <= 1 - val_frac, "train", "val")
    return out


def check_no_time_leak(
    df: pd.DataFrame, split: pd.Series, group_cols: Sequence[str] = ("day", "label")
) -> None:
    d = df[[*group_cols, "Timestamp"]].assign(split=split.to_numpy())
    agg = d.groupby([*group_cols, "split"], observed=True)["Timestamp"].agg(["min", "max"])
    problems: list[str] = []
    for key, grp in agg.groupby(level=list(range(len(group_cols)))):
        by_split = grp.droplevel(list(range(len(group_cols))))
        tr_max = by_split["max"].get("train")
        va_max = by_split["max"].get("val")
        for later in ("val", "test"):
            lo = by_split["min"].get(later)
            if tr_max is not None and lo is not None and lo < tr_max:
                problems.append(f"{key}: {later} starts {lo} before train ends {tr_max}")
        te_min = by_split["min"].get("test")
        if va_max is not None and te_min is not None and te_min < va_max:
            problems.append(f"{key}: test starts {te_min} before val ends {va_max}")
    if problems:
        raise LeakError("; ".join(problems[:10]))


def split_report(df: pd.DataFrame, split: pd.Series, name: str) -> dict[str, Any]:
    d = df.assign(split=split.to_numpy())
    try:
        check_no_time_leak(df, split)
        leak = "ok"
    except LeakError as e:
        leak = f"LEAK: {e}"
    counts = d.groupby(["split", "family"], observed=True).size()
    by_split_family = {
        sp: {str(f): int(n) for (s2, f), n in counts.items() if s2 == sp}
        for sp in ("train", "val", "test")
    }
    ranges = d.groupby(["split", "day"], observed=True)["Timestamp"].agg(["min", "max"])
    active_idle_cols = [c for c in s.FEATURE_COLS if c.startswith(("Active", "Idle"))]
    active_idle_max_seconds = float(np.nanmax(df[active_idle_cols].to_numpy()) / 1e6)
    return {
        "name": name,
        "rows": int(len(df)),
        "leak_check": leak,
        "counts_by_split": {str(k): int(v) for k, v in d["split"].value_counts().items()},
        "counts_by_split_family": by_split_family,
        "time_ranges": {
            f"{sp}/{day}": {"min": str(r["min"]), "max": str(r["max"])}
            for (sp, day), r in ranges.iterrows()
        },
        "active_idle_max_seconds": active_idle_max_seconds,
    }
```

- [ ] **Step 4: Implement `build.py`, wire the CLI, write `configs/data.yaml`**

`src/secops/data/build.py`:
```python
"""End-to-end build: raw CSVs -> cleaned, split Parquet + JSON reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from secops.data.clean import AttemptedPolicy, clean
from secops.data.ingest import read_all
from secops.data.manifest import manifest_digest
from secops.data.split import (
    SPLIT_COLUMN,
    SplitStrategy,
    assign_chrono_within_group,
    assign_heldout_day,
    check_no_time_leak,
    split_report,
)

log = logging.getLogger(__name__)


def build(
    raw_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    policy: AttemptedPolicy,
    subdir: str = "improved",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Path:
    log.info("reading raw CSVs from %s", raw_dir / subdir)
    df = read_all(raw_dir, subdir=subdir)
    log.info("cleaning %d rows with policy=%s", len(df), policy)
    df, cleaning = clean(df, policy)
    chrono = assign_chrono_within_group(df, train_frac=train_frac, val_frac=val_frac)
    heldout = assign_heldout_day(df, test_day="friday", val_frac=val_frac)
    check_no_time_leak(df, chrono)
    check_no_time_leak(df, heldout)
    df[SPLIT_COLUMN[SplitStrategy.CHRONO_WITHIN_GROUP]] = chrono.astype("category")
    df[SPLIT_COLUMN[SplitStrategy.HELDOUT_FRIDAY]] = heldout.astype("category")

    out_dir = processed_dir / str(policy)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "flows.parquet"
    df.to_parquet(out, index=False)

    rep_dir = reports_dir / str(policy)
    cleaning.to_json(rep_dir / "cleaning_report.json")
    reports = {
        "manifest_digest": manifest_digest(),
        "attempted_policy": str(policy),
        "chrono_within_group": split_report(df, chrono, "chrono_within_group"),
        "heldout_friday": split_report(df, heldout, "heldout_friday"),
    }
    (rep_dir / "split_report.json").write_text(json.dumps(reports, indent=2))
    log.info("wrote %s (%d rows)", out, len(df))
    return out
```

Add to `src/secops/data/cli.py` (new imports: `import logging`, `from secops.config import Settings`, `from secops.data.build import build as build_pipeline`, `from secops.data.clean import AttemptedPolicy`):
```python
@app.command()
def build(
    attempted_policy: AttemptedPolicy = AttemptedPolicy.RELABEL_BENIGN,
    data_dir: Path | None = None,
    subdir: str = "improved",
) -> None:
    """Clean, split and write processed/<policy>/flows.parquet with reports."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    st = get_settings() if data_dir is None else Settings(data_dir=data_dir)
    out = build_pipeline(
        st.raw_dir, st.processed_dir, st.reports_dir, attempted_policy, subdir=subdir
    )
    typer.echo(f"built {out}")
```

`configs/data.yaml` (documentation of defaults; the CLI flags are the source of truth):
```yaml
attempted_policy: relabel_benign
train_frac: 0.70
val_frac: 0.15
heldout_test_day: friday
max_fpr: 0.01
```

- [ ] **Step 5: Write the integration test**

`tests/integration/test_build_pipeline.py`:
```python
import json
from pathlib import Path

import pandas as pd
import pytest

from secops.data import schema as s
from secops.data.build import build
from secops.data.clean import AttemptedPolicy

pytestmark = pytest.mark.integration


def test_build_writes_parquet_and_reports(fixture_dir: Path, tmp_path: Path) -> None:
    out = build(
        fixture_dir,
        tmp_path / "processed",
        tmp_path / "reports",
        AttemptedPolicy.RELABEL_BENIGN,
        subdir="",
    )
    df = pd.read_parquet(out)
    expected = [
        "id",
        "day",
        *s.META_COLS,
        *s.FEATURE_COLS,
        "label_raw",
        "label",
        "family",
        "is_attack",
        "split_chrono",
        "split_heldout",
    ]
    for c in expected:
        assert c in df.columns, c
    rep = json.loads((tmp_path / "reports" / "relabel_benign" / "split_report.json").read_text())
    assert rep["chrono_within_group"]["leak_check"] == "ok"
    assert rep["heldout_friday"]["leak_check"] == "ok"
    cl = json.loads((tmp_path / "reports" / "relabel_benign" / "cleaning_report.json").read_text())
    assert cl["rows_out"] == len(df)
```

- [ ] **Step 6: Run unit + integration tests, then the full-data build**

Run: `uv run pytest tests/unit/data/test_split.py tests/integration -m integration -v`
Expected: 7 passed

Run (full data, takes minutes; watch memory with `free -h` in another shell):
```bash
export SECOPS_DATA_DIR=$HOME/data/secops
uv run secops-data build --attempted-policy relabel_benign
uv run secops-data build --attempted-policy drop
python3 -c "import json;r=json.load(open('$HOME/data/secops/reports/relabel_benign/cleaning_report.json'));print(r['rows_out'], r['duplicates_removed'], r['inf_cells_replaced'])"
```
Expected: two Parquet files; both split reports show `"leak_check": "ok"`; `inf_cells_replaced` is 8; record `rows_out` and `duplicates_removed` (the TBD numbers in Part A) for `docs/dataset.md`.

- [ ] **Step 7: Commit**

```bash
git add src/secops/data/split.py src/secops/data/build.py src/secops/data/cli.py configs/data.yaml tests/unit/data/test_split.py tests/integration
git commit -m "feat(data): chronological splits with leak check and build pipeline

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Metrics and threshold selection

**Files:**
- Create: `src/secops/detection/__init__.py`, `src/secops/detection/metrics.py`, `tests/unit/detection/__init__.py`, `tests/unit/detection/test_metrics.py`

**Interfaces:**
- Produces: `@dataclass BinaryMetrics(pr_auc, roc_auc, brier, threshold, precision, recall, f1, fpr, accuracy, tp, fp, tn, fn)` with `to_dict() -> dict[str, float]`; `binary_metrics(y_true, y_prob, threshold) -> BinaryMetrics`; `@dataclass ThresholdChoice(threshold, fpr, recall, method)`; `select_threshold(y_true, y_prob, max_fpr=0.01) -> ThresholdChoice` (methods `"max_recall_at_fpr"` or `"fallback_max_threshold"`); `max_f1_threshold(y_true, y_prob) -> ThresholdChoice` (method `"max_f1"`); `recall_at_fprs(y_true, y_prob, fprs=(0.001,0.005,0.01,0.05)) -> dict[str,float]` keyed `recall_at_fpr_<f>`; `multiclass_metrics(y_true, y_pred, y_prob, classes) -> dict` with keys `macro_f1, weighted_f1, log_loss, per_class, confusion`; `per_group_recall(y_true, y_prob, threshold, groups) -> dict[str, float]`.

- [ ] **Step 1: Write failing tests**

`tests/unit/detection/test_metrics.py`:
```python
import numpy as np
import pytest

from secops.detection.metrics import (
    binary_metrics,
    max_f1_threshold,
    multiclass_metrics,
    per_group_recall,
    recall_at_fprs,
    select_threshold,
)

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
P = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])


def test_binary_metrics_hand_computed() -> None:
    m = binary_metrics(Y, P, threshold=0.5)
    # >= 0.5: idx 2 (FP), 4, 5, 7 (TP); idx 6 is FN
    assert (m.tp, m.fp, m.tn, m.fn) == (3, 1, 3, 1)
    assert m.precision == pytest.approx(0.75)
    assert m.recall == pytest.approx(0.75)
    assert m.fpr == pytest.approx(0.25)
    assert m.accuracy == pytest.approx(0.75)
    assert 0 < m.pr_auc <= 1 and 0 < m.roc_auc <= 1
    assert set(m.to_dict()) >= {"pr_auc", "roc_auc", "recall", "fpr", "threshold"}


def test_select_threshold_respects_fpr_budget() -> None:
    c = select_threshold(Y, P, max_fpr=0.0)
    m = binary_metrics(Y, P, c.threshold)
    assert m.fpr == 0.0
    assert c.method == "max_recall_at_fpr"
    assert c.recall == pytest.approx(m.recall)


def test_select_threshold_picks_highest_recall_under_budget() -> None:
    c = select_threshold(Y, P, max_fpr=0.25)
    assert binary_metrics(Y, P, c.threshold).fpr <= 0.25
    assert c.recall >= select_threshold(Y, P, max_fpr=0.0).recall


def test_select_threshold_fallback_when_no_threshold_meets_fpr() -> None:
    y = np.array([0, 1])
    p = np.array([0.9, 0.9])  # catching the positive always catches the negative
    c = select_threshold(y, p, max_fpr=0.0)
    assert c.method == "fallback_max_threshold"
    assert binary_metrics(y, p, c.threshold).fp == 0


def test_max_f1_threshold_and_recall_at_fprs() -> None:
    c = max_f1_threshold(Y, P)
    assert 0 < c.threshold <= 1 and c.method == "max_f1"
    r = recall_at_fprs(Y, P, fprs=(0.0, 0.25, 1.0))
    assert r["recall_at_fpr_1.0"] == 1.0
    assert r["recall_at_fpr_0.0"] <= r["recall_at_fpr_0.25"] <= 1.0


def test_multiclass_metrics_shape() -> None:
    classes = ["a", "b", "c"]
    yt = np.array(["a", "b", "c", "a"])
    yp = np.array(["a", "b", "a", "a"])
    prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2], [0.7, 0.2, 0.1]])
    m = multiclass_metrics(yt, yp, prob, classes)
    assert set(m["per_class"]) == set(classes)
    assert m["per_class"]["a"]["support"] == 2
    assert len(m["confusion"]) == 3 and len(m["confusion"][0]) == 3
    assert 0 <= m["macro_f1"] <= 1 and m["log_loss"] > 0


def test_per_group_recall() -> None:
    groups = np.array(["x", "x", "y", "y", "x", "x", "y", "y"])
    r = per_group_recall(Y, P, 0.5, groups)
    assert r["x"] == pytest.approx(1.0)  # positives in x: idx 4, 5 -> both >= 0.5
    assert r["y"] == pytest.approx(0.5)  # positives in y: idx 6 (0.4), 7 (0.7)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/detection/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.detection'`

- [ ] **Step 3: Implement `metrics.py`** (`src/secops/detection/__init__.py` is a one-line docstring)

```python
"""Detection metrics. PR-AUC first; accuracy is reported but never a headline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    brier: float
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass
class ThresholdChoice:
    threshold: float
    fpr: float
    recall: float
    method: str


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_metrics(y_true: ArrayLike, y_prob: ArrayLike, threshold: float) -> BinaryMetrics:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    single_class = len(np.unique(y)) < 2
    return BinaryMetrics(
        pr_auc=float("nan") if single_class else float(average_precision_score(y, p)),
        roc_auc=float("nan") if single_class else float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=_safe_div(fp, fp + tn),
        accuracy=_safe_div(tp + tn, len(y)),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def _strictest(p: np.ndarray) -> ThresholdChoice:
    t = float(np.nextafter(p.max(), np.inf))
    return ThresholdChoice(t, 0.0, 0.0, "fallback_max_threshold")


def select_threshold(
    y_true: ArrayLike, y_prob: ArrayLike, max_fpr: float = 0.01
) -> ThresholdChoice:
    """Highest recall whose FPR <= max_fpr; ties -> higher threshold; fallback if none qualifies."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    fpr, tpr, thr = roc_curve(y, p)
    ok = fpr <= max_fpr
    if not ok.any():
        return _strictest(p)
    best_tpr = tpr[ok].max()
    cands = np.where(ok & (tpr == best_tpr))[0]
    i = cands[np.argmax(thr[cands])]
    t = float(min(thr[i], 1.0))  # roc_curve's first threshold is +inf
    m = binary_metrics(y, p, t)
    if m.fpr > max_fpr or m.recall == 0.0:
        return _strictest(p)
    return ThresholdChoice(t, m.fpr, m.recall, "max_recall_at_fpr")


def max_f1_threshold(y_true: ArrayLike, y_prob: ArrayLike) -> ThresholdChoice:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    grid = np.unique(p)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    t = float(grid[int(np.argmax(scores))])
    m = binary_metrics(y, p, t)
    return ThresholdChoice(t, m.fpr, m.recall, "max_f1")


def recall_at_fprs(
    y_true: ArrayLike, y_prob: ArrayLike, fprs: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)
) -> dict[str, float]:
    return {f"recall_at_fpr_{f}": select_threshold(y_true, y_prob, f).recall for f in fprs}


def multiclass_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, y_prob: ArrayLike, classes: list[str]
) -> dict[str, Any]:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=classes, zero_division=0)
    per_class = {
        c: {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
        for i, c in enumerate(classes)
    }
    return {
        "macro_f1": float(f1_score(yt, yp, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, labels=classes, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(yt, np.asarray(y_prob), labels=classes)),
        "per_class": per_class,
        "confusion": confusion_matrix(yt, yp, labels=classes).tolist(),
    }


def per_group_recall(
    y_true: ArrayLike, y_prob: ArrayLike, threshold: float, groups: ArrayLike
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    g = np.asarray(groups).astype(str)
    out: dict[str, float] = {}
    for name in np.unique(g):
        mask = (g == name) & (y == 1)
        if mask.any():
            out[str(name)] = float(pred[mask].mean())
    return out
```

Note: `test_select_threshold_fallback_when_no_threshold_meets_fpr` passes because with `y=[0,1]`, `p=[0.9,0.9]` the only finite operating point has FPR 1.0; `roc_curve` also yields the `+inf` point with FPR 0 and TPR 0, which the `m.recall == 0.0` guard routes to the fallback.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/detection/test_metrics.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/secops/detection tests/unit/detection
git commit -m "feat(detection): binary/multiclass metrics and FPR-budget threshold selection

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Feature spec and model factory

**Files:**
- Create: `src/secops/detection/features.py`, `src/secops/detection/models.py`, `tests/unit/detection/test_features.py`, `tests/unit/detection/test_models.py`

**Interfaces:**
- Produces: `class FeatureSpec(BaseModel)` with `names: list[str]`, `version: str`; `to_matrix(df, strict_order=False) -> np.ndarray` (float32; `ValueError` mentioning "missing" for absent columns, "order" when `strict_order` and the frame's feature columns are not in spec order); `save(path) -> Path`; `load(path) -> FeatureSpec`; `FEATURE_SPEC_V1` (version `"v1-noport"`), `FEATURE_SPEC_V1_WITHPORT` (version `"v1-withport"`), `FEATURE_SPECS: dict[str, FeatureSpec]` keyed by version.
- `ModelName = Literal["logreg","xgboost","lightgbm"]`, `Task = Literal["binary","multiclass"]`, `Weighting = Literal["none","balanced"]`, `DEFAULT_PARAMS`; `build_model(name, task, params, seed, n_classes=None) -> Any`; `fit_model(model, X_train, y_train, X_val, y_val, weighting, name) -> Any`; `predict_proba_positive(model, X) -> np.ndarray`.

- [ ] **Step 1: Write failing tests**

`tests/unit/detection/test_features.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secops.data import schema as s
from secops.detection.features import FEATURE_SPEC_V1, FEATURE_SPEC_V1_WITHPORT, FeatureSpec


def _frame(cols: list[str], n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(np.ones((n, len(cols)), dtype=np.float32), columns=cols)


def test_v1_spec_matches_schema() -> None:
    assert FEATURE_SPEC_V1.names == s.FEATURE_COLS
    assert FEATURE_SPEC_V1.version == "v1-noport"
    assert FEATURE_SPEC_V1_WITHPORT.names[-1] == s.PORT_COL


def test_to_matrix_selects_in_spec_order_from_wider_frame() -> None:
    df = _frame([*reversed(s.FEATURE_COLS), "label", "Dst Port"])
    X = FEATURE_SPEC_V1.to_matrix(df)
    assert X.shape == (3, 82) and X.dtype == np.float32


def test_feature_spec_rejects_reordered_columns() -> None:
    df = _frame(list(reversed(s.FEATURE_COLS)))
    with pytest.raises(ValueError, match="order"):
        FEATURE_SPEC_V1.to_matrix(df, strict_order=True)


def test_to_matrix_rejects_missing_column() -> None:
    df = _frame(s.FEATURE_COLS[:-1])
    with pytest.raises(ValueError, match="missing"):
        FEATURE_SPEC_V1.to_matrix(df)


def test_save_load_round_trip(tmp_path: Path) -> None:
    p = FEATURE_SPEC_V1.save(tmp_path / "fs.json")
    assert FeatureSpec.load(p) == FEATURE_SPEC_V1
```

`tests/unit/detection/test_models.py`:
```python
import numpy as np
import pytest

from secops.detection.models import build_model, fit_model, predict_proba_positive

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 10)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
X[::50, 2] = np.nan  # NaN cells like inf->nan rate features
Xv, yv = X[:100], y[:100]


@pytest.mark.parametrize("name", ["logreg", "xgboost", "lightgbm"])
@pytest.mark.parametrize("weighting", ["none", "balanced"])
def test_binary_models_fit_and_score(name: str, weighting: str) -> None:
    m = build_model(name, "binary", {}, seed=1)  # type: ignore[arg-type]
    m = fit_model(m, X, y, Xv, yv, weighting=weighting, name=name)  # type: ignore[arg-type]
    p = predict_proba_positive(m, Xv)
    assert p.shape == (100,) and (0 <= p).all() and (p <= 1).all()
    assert ((p >= 0.5).astype(int) == yv).mean() > 0.8


def test_logreg_pipeline_handles_nan() -> None:
    m = build_model("logreg", "binary", {}, seed=1)
    m = fit_model(m, X, y, Xv, yv, weighting="none", name="logreg")
    assert np.isfinite(predict_proba_positive(m, Xv)).all()


def test_logreg_multiclass_not_supported() -> None:
    with pytest.raises(ValueError):
        build_model("logreg", "multiclass", {}, seed=1, n_classes=3)


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_multiclass_models(name: str) -> None:
    ym = np.digitize(X[:, 0], [-0.5, 0.5])  # classes 0, 1, 2
    m = build_model(name, "multiclass", {}, seed=1, n_classes=3)  # type: ignore[arg-type]
    m = fit_model(m, X, ym, Xv, ym[:100], weighting="balanced", name=name)  # type: ignore[arg-type]
    proba = m.predict_proba(Xv)
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/detection/test_features.py tests/unit/detection/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `features.py`**

```python
"""Feature specification: the exact ordered list of columns a model consumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from secops.data.schema import FEATURE_COLS, PORT_COL


class FeatureSpec(BaseModel):
    names: list[str]
    version: str

    def to_matrix(self, df: pd.DataFrame, strict_order: bool = False) -> np.ndarray:
        missing = [c for c in self.names if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:10]}")
        if strict_order:
            wanted = set(self.names)
            present = [c for c in df.columns if c in wanted]
            if present != self.names:
                raise ValueError("feature columns are not in spec order")
        return df[self.names].to_numpy(dtype=np.float32)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> FeatureSpec:
        return cls.model_validate_json(path.read_text())


FEATURE_SPEC_V1 = FeatureSpec(names=list(FEATURE_COLS), version="v1-noport")
FEATURE_SPEC_V1_WITHPORT = FeatureSpec(names=[*FEATURE_COLS, PORT_COL], version="v1-withport")
FEATURE_SPECS: dict[str, FeatureSpec] = {
    fs.version: fs for fs in (FEATURE_SPEC_V1, FEATURE_SPEC_V1_WITHPORT)
}
```

- [ ] **Step 4: Implement `models.py`**

```python
"""Model factory and fitting helpers for the three baselines."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ModelName = Literal["logreg", "xgboost", "lightgbm"]
Task = Literal["binary", "multiclass"]
Weighting = Literal["none", "balanced"]

EARLY_STOPPING_ROUNDS = 50

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
    "xgboost": {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    },
    "lightgbm": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    },
}


def build_model(
    name: ModelName, task: Task, params: dict[str, Any], seed: int, n_classes: int | None = None
) -> Any:
    p = {**DEFAULT_PARAMS[name], **params}
    if name == "logreg":
        if task == "multiclass":
            raise ValueError("logreg is a binary-only baseline in Phase 1")
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(random_state=seed, **p)),
            ]
        )
    if name == "xgboost":
        if task == "binary":
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=seed,
                n_jobs=-1,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                **p,
            )
        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **p,
        )
    if name == "lightgbm":
        objective = "binary" if task == "binary" else "multiclass"
        return LGBMClassifier(objective=objective, random_state=seed, n_jobs=-1, verbose=-1, **p)
    raise ValueError(name)


def fit_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weighting: Weighting,
    name: ModelName,
) -> Any:
    sw = compute_sample_weight("balanced", y_train) if weighting == "balanced" else None
    if name == "logreg":
        model.fit(X_train, y_train, clf__sample_weight=sw)
    elif name == "xgboost":
        model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "lightgbm":
        metric = "average_precision" if len(np.unique(y_train)) == 2 else "multi_logloss"
        model.fit(
            X_train,
            y_train,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            eval_metric=metric,
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
        )
    else:
        raise ValueError(name)
    return model


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/detection/test_features.py tests/unit/detection/test_models.py -v`
Expected: 5 + 10 passed. If the installed LightGBM rejects `average_precision` as an eval metric name, switch to `"auc"` and say so in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/secops/detection/features.py src/secops/detection/models.py tests/unit/detection/test_features.py tests/unit/detection/test_models.py
git commit -m "feat(detection): feature spec and logreg/xgboost/lightgbm model factory

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: SHAP explainer and diagnostic plots

**Files:**
- Create: `src/secops/detection/explain.py`, `src/secops/detection/plots.py`, `tests/unit/detection/test_explain.py`, `tests/unit/detection/test_plots.py`

Load the `dataviz` skill before writing `plots.py`; keep the figures minimal and deterministic.

**Interfaces:**
- `make_explainer(model, name: ModelName, X_background) -> Any` (object with `shap_values(X)`); `global_importance(explainer, X, feature_names) -> pd.DataFrame` with columns `feature, mean_abs_shap` sorted descending; `@dataclass FeatureContribution(feature, value, shap_value)`; `top_k_contributions(explainer, x, feature_names, k=5) -> list[FeatureContribution]` sorted by |shap| descending.
- `pr_curve_figure(y_true, y_prob, threshold) -> Figure`; `confusion_matrix_figure(cm, labels) -> Figure`; `shap_bar_figure(importance, top_n=20) -> Figure`; `save_figure(fig, path) -> Path`.

- [ ] **Step 1: Write failing tests**

`tests/unit/detection/test_explain.py`:
```python
from typing import Any

import numpy as np

from secops.detection.explain import global_importance, make_explainer, top_k_contributions
from secops.detection.models import build_model, fit_model

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 6)).astype(np.float32)
y = (2 * X[:, 0] - X[:, 3] > 0).astype(int)
names = [f"f{i}" for i in range(6)]


def _fit(name: Any) -> Any:
    params = {"n_estimators": 50} if name != "logreg" else {}
    m = build_model(name, "binary", params, seed=0)
    return fit_model(m, X, y, X[:50], y[:50], weighting="none", name=name)


def test_global_importance_ranks_true_drivers_first() -> None:
    for name in ("lightgbm", "xgboost", "logreg"):
        ex = make_explainer(_fit(name), name, X[:100])  # type: ignore[arg-type]
        imp = global_importance(ex, X[:100], names)
        assert list(imp.columns) == ["feature", "mean_abs_shap"]
        assert set(imp["feature"].head(2)) == {"f0", "f3"}, name


def test_top_k_contributions_sorted_by_magnitude() -> None:
    ex = make_explainer(_fit("lightgbm"), "lightgbm", X[:100])
    top = top_k_contributions(ex, X[0], names, k=3)
    assert len(top) == 3
    mags = [abs(c.shap_value) for c in top]
    assert mags == sorted(mags, reverse=True)
    assert top[0].feature in {"f0", "f3"}
```

`tests/unit/detection/test_plots.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd

from secops.detection.plots import (
    confusion_matrix_figure,
    pr_curve_figure,
    save_figure,
    shap_bar_figure,
)


def test_figures_render_and_save(tmp_path: Path) -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    assert save_figure(pr_curve_figure(y, p, 0.5), tmp_path / "pr.png").stat().st_size > 0
    cm = confusion_matrix_figure([[2, 0], [1, 1]], ["benign", "attack"])
    assert save_figure(cm, tmp_path / "cm.png").exists()
    imp = pd.DataFrame({"feature": ["a", "b"], "mean_abs_shap": [0.5, 0.2]})
    assert save_figure(shap_bar_figure(imp), tmp_path / "shap.png").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/detection/test_explain.py tests/unit/detection/test_plots.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `explain.py`**

```python
"""SHAP explanations for the detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from secops.detection.models import ModelName


@dataclass
class FeatureContribution:
    feature: str
    value: float
    shap_value: float


class _PipelineExplainer:
    """LinearExplainer over the fitted imputer+scaler of the logistic-regression pipeline."""

    def __init__(self, transform: Any, inner: Any) -> None:
        self.transform, self.inner = transform, inner

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.inner.shap_values(self.transform.transform(X)))


def make_explainer(model: Any, name: ModelName, X_background: np.ndarray) -> Any:
    if name in ("xgboost", "lightgbm"):
        return shap.TreeExplainer(model)
    transform = model[:-1]
    clf = model[-1]
    return _PipelineExplainer(
        transform, shap.LinearExplainer(clf, transform.transform(X_background))
    )


def _positive_class_shap(explainer: Any, X: np.ndarray) -> np.ndarray:
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older shap API for binary tree models: [neg, pos]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        sv = sv[:, :, -1]
    return sv


def global_importance(explainer: Any, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    sv = _positive_class_shap(explainer, X)
    imp = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_k_contributions(
    explainer: Any, x: np.ndarray, feature_names: list[str], k: int = 5
) -> list[FeatureContribution]:
    sv = _positive_class_shap(explainer, x.reshape(1, -1))[0]
    order = np.argsort(-np.abs(sv))[:k]
    return [FeatureContribution(feature_names[i], float(x[i]), float(sv[i])) for i in order]
```

- [ ] **Step 4: Implement `plots.py`**

```python
"""Diagnostic figures logged to MLflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


def pr_curve_figure(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Figure:
    p, r, t = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, lw=1.5)
    i = min(int(np.searchsorted(t, threshold)), len(r) - 1) if len(t) else 0
    ax.scatter([r[i]], [p[i]], zorder=3, label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def confusion_matrix_figure(cm: list[list[int]], labels: list[str]) -> Figure:
    m = np.asarray(cm)
    size = 1.2 * len(labels) + 2
    fig, ax = plt.subplots(figsize=(size, size - 0.5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def shap_bar_figure(importance: pd.DataFrame, top_n: int = 20) -> Figure:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(top) + 1.5))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP value|")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/detection/test_explain.py tests/unit/detection/test_plots.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/secops/detection/explain.py src/secops/detection/plots.py tests/unit/detection/test_explain.py tests/unit/detection/test_plots.py
git commit -m "feat(detection): SHAP explainer helpers and diagnostic figures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Training pipeline with MLflow logging and the `secops-train` CLI

**Files:**
- Create: `src/secops/detection/train.py`, `src/secops/detection/cli.py`, `configs/models/logreg_binary.yaml`, `configs/models/xgboost_binary.yaml`, `configs/models/lightgbm_binary.yaml`, `tests/integration/test_train_pipeline.py`

**Interfaces:**
- `class TrainConfig(BaseModel)`: `experiment: str`, `run_name: str`, `task: Task`, `model: ModelName`, `weighting: Weighting = "balanced"`, `params: dict[str, Any] = {}`, `feature_spec: str = "v1-noport"`, `split_strategy: SplitStrategy = CHRONO_WITHIN_GROUP`, `attempted_policy: AttemptedPolicy = RELABEL_BENIGN`, `max_fpr: float = 0.01`, `shap_sample: int = 20000`, `reuse_threshold_from_run: str | None = None`, `register_as: str | None = None`, `seed: int = 42`; `TrainConfig.from_yaml(path, overrides=None)`.
- `load_parts(processed_dir, policy, split_col) -> dict[str, pd.DataFrame]` keyed `train/val/test`.
- `run_training(cfg, settings) -> str` (MLflow run id). Logs tags, params, metrics and artifacts per A13/A14. Multiclass runs log `per_class_metrics.json` and `confusion_matrix.png` but no `pr_curve.png`, `threshold.json` or `per_label_recall.csv`.
- CLI: `secops-train run --config PATH [--set key=value ...]`, `secops-train report --experiment NAME`, and (Task 11) `secops-train promote-best`.

- [ ] **Step 1: Write the YAML configs**

`configs/models/lightgbm_binary.yaml`:
```yaml
experiment: secops/detection-binary
run_name: lightgbm_balanced
task: binary
model: lightgbm
weighting: balanced
feature_spec: v1-noport
split_strategy: chrono_within_group
attempted_policy: relabel_benign
max_fpr: 0.01
shap_sample: 20000
```
`xgboost_binary.yaml` and `logreg_binary.yaml`: identical except `model` and `run_name` (`xgboost_balanced`, `logreg_balanced`). The `weighting: none` variants are produced with `--set weighting=none --set run_name=<model>_none`. Registration is done by `promote-best` in Task 11, not by the configs.

- [ ] **Step 2: Write the failing integration test**

`tests/integration/test_train_pipeline.py`:
```python
from pathlib import Path

import mlflow
import pytest

from secops.config import Settings
from secops.data.build import build
from secops.data.clean import AttemptedPolicy
from secops.detection.train import TrainConfig, run_training

pytestmark = pytest.mark.integration


@pytest.fixture
def built_settings(fixture_dir: Path, tmp_path: Path) -> Settings:
    st = Settings(data_dir=tmp_path, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    build(fixture_dir, st.processed_dir, st.reports_dir, AttemptedPolicy.RELABEL_BENIGN, subdir="")
    return st


def _artifact_paths(run_id: str) -> set[str]:
    return {a.path for a in mlflow.artifacts.list_artifacts(run_id=run_id)}


def test_binary_training_logs_metrics_and_artifacts(built_settings: Settings) -> None:
    cfg = TrainConfig(
        experiment="test/binary",
        run_name="lgbm",
        task="binary",
        model="lightgbm",
        weighting="balanced",
        params={"n_estimators": 30},
        shap_sample=50,
    )
    run_id = run_training(cfg, built_settings)
    mlflow.set_tracking_uri(built_settings.resolved_tracking_uri())
    run = mlflow.get_run(run_id)
    for key in (
        "val_pr_auc",
        "test_pr_auc",
        "test_recall",
        "test_fpr",
        "test_precision",
        "threshold",
    ):
        assert key in run.data.metrics, key
    assert run.data.tags["attempted_policy"] == "relabel_benign"
    assert run.data.tags["feature_spec"] == "v1-noport"
    arts = _artifact_paths(run_id)
    for a in (
        "metrics.json",
        "threshold.json",
        "feature_spec.json",
        "pr_curve.png",
        "confusion_matrix.png",
        "per_label_recall.csv",
        "shap_global_importance.csv",
        "shap_summary.png",
        "explainer_background.parquet",
        "model",
    ):
        assert a in arts, a


def test_family_training_logs_per_class(built_settings: Settings) -> None:
    cfg = TrainConfig(
        experiment="test/family",
        run_name="lgbm_family",
        task="multiclass",
        model="lightgbm",
        weighting="balanced",
        params={"n_estimators": 30},
        shap_sample=50,
    )
    run_id = run_training(cfg, built_settings)
    mlflow.set_tracking_uri(built_settings.resolved_tracking_uri())
    run = mlflow.get_run(run_id)
    assert "test_macro_f1" in run.data.metrics
    assert any(k.startswith("test_recall_") for k in run.data.metrics)
    arts = _artifact_paths(run_id)
    assert "per_class_metrics.json" in arts and "confusion_matrix.png" in arts
    assert "threshold.json" not in arts


def test_logreg_training_runs(built_settings: Settings) -> None:
    cfg = TrainConfig(
        experiment="test/binary",
        run_name="lr",
        task="binary",
        model="logreg",
        weighting="none",
        shap_sample=50,
    )
    run_id = run_training(cfg, built_settings)
    mlflow.set_tracking_uri(built_settings.resolved_tracking_uri())
    assert "test_pr_auc" in mlflow.get_run(run_id).data.metrics
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/integration/test_train_pipeline.py -m integration -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.detection.train'`

- [ ] **Step 4: Implement `train.py`**

```python
"""Training runs: load split Parquet, fit, evaluate, log to MLflow."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from secops.config import Settings
from secops.data import schema as s
from secops.data.clean import AttemptedPolicy
from secops.data.manifest import manifest_digest
from secops.data.split import SPLIT_COLUMN, SplitStrategy
from secops.detection.explain import global_importance, make_explainer
from secops.detection.features import FEATURE_SPECS, FeatureSpec
from secops.detection.metrics import (
    binary_metrics,
    max_f1_threshold,
    multiclass_metrics,
    per_group_recall,
    recall_at_fprs,
    select_threshold,
)
from secops.detection.models import (
    ModelName,
    Task,
    Weighting,
    build_model,
    fit_model,
    predict_proba_positive,
)
from secops.detection.plots import (
    confusion_matrix_figure,
    pr_curve_figure,
    save_figure,
    shap_bar_figure,
)

log = logging.getLogger(__name__)
BACKGROUND_ROWS = 1000


class TrainConfig(BaseModel):
    experiment: str
    run_name: str
    task: Task
    model: ModelName
    weighting: Weighting = "balanced"
    params: dict[str, Any] = Field(default_factory=dict)
    feature_spec: str = "v1-noport"
    split_strategy: SplitStrategy = SplitStrategy.CHRONO_WITHIN_GROUP
    attempted_policy: AttemptedPolicy = AttemptedPolicy.RELABEL_BENIGN
    max_fpr: float = 0.01
    shap_sample: int = 20000
    reuse_threshold_from_run: str | None = None
    register_as: str | None = None
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
        data = yaml.safe_load(path.read_text()) or {}
        for k, v in (overrides or {}).items():
            data[k] = None if v == "" else v
        return cls.model_validate(data)


def load_parts(
    processed_dir: Path, policy: AttemptedPolicy, split_col: str
) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(processed_dir / str(policy) / "flows.parquet")
    return {name: df[df[split_col] == name] for name in ("train", "val", "test")}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603, S607
    except Exception:  # noqa: BLE001
        return "unknown"


def _select_rows(df: pd.DataFrame, task: Task) -> pd.DataFrame:
    if task == "binary":
        return df
    fam = df["family"].astype(str)
    return df[(df["is_attack"] == 1) & fam.isin(s.FAMILY_CLASSIFIER_CLASSES)]


def _targets(df: pd.DataFrame, task: Task) -> np.ndarray:
    if task == "binary":
        return df["is_attack"].to_numpy(dtype=int)
    return df["family"].astype(str).to_numpy()


def _log_model(model: Any, name: ModelName, X_example: np.ndarray) -> None:
    sig = mlflow.models.infer_signature(X_example, model.predict_proba(X_example))
    if name == "xgboost":
        mlflow.xgboost.log_model(model, "model", signature=sig, input_example=X_example[:5])
    elif name == "lightgbm":
        mlflow.lightgbm.log_model(model, "model", signature=sig, input_example=X_example[:5])
    else:
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=sig,
            input_example=X_example[:5],
            pyfunc_predict_fn="predict_proba",
        )


def _binary_stage(
    cfg: TrainConfig,
    model: Any,
    X: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    parts: dict[str, pd.DataFrame],
    out: Path,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    p_val = predict_proba_positive(model, X["val"])
    p_test = predict_proba_positive(model, X["test"])
    if cfg.reuse_threshold_from_run:
        prev = json.loads(
            mlflow.artifacts.load_text(f"runs:/{cfg.reuse_threshold_from_run}/threshold.json")
        )
        choice: dict[str, Any] = {
            "threshold": prev["threshold"],
            "method": f"reused:{cfg.reuse_threshold_from_run}",
        }
    else:
        c = select_threshold(y["val"], p_val, cfg.max_fpr)
        choice = {
            "threshold": c.threshold,
            "method": c.method,
            "val_fpr": c.fpr,
            "val_recall": c.recall,
            "max_f1_threshold": max_f1_threshold(y["val"], p_val).threshold,
        }
    t = float(choice["threshold"])
    for part, p in (("val", p_val), ("test", p_test)):
        metrics.update(
            {f"{part}_{k}": v for k, v in binary_metrics(y[part], p, t).to_dict().items()}
        )
        metrics.update({f"{part}_{k}": v for k, v in recall_at_fprs(y[part], p).items()})
    metrics["threshold"] = t
    (out / "threshold.json").write_text(json.dumps(choice, indent=2))
    labels = parts["test"]["label"].astype(str).to_numpy()
    per_label = per_group_recall(y["test"], p_test, t, labels)
    pd.Series(per_label, name="recall").rename_axis("label").to_csv(out / "per_label_recall.csv")
    save_figure(pr_curve_figure(y["test"], p_test, t), out / "pr_curve.png")
    cm = [
        [int(metrics["test_tn"]), int(metrics["test_fp"])],
        [int(metrics["test_fn"]), int(metrics["test_tp"])],
    ]
    save_figure(confusion_matrix_figure(cm, ["benign", "attack"]), out / "confusion_matrix.png")
    return metrics


def _multiclass_stage(
    model: Any, X: dict[str, np.ndarray], y: dict[str, np.ndarray], classes: list[str], out: Path
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for part in ("val", "test"):
        proba = np.asarray(model.predict_proba(X[part]))
        pred = np.asarray(classes)[proba.argmax(axis=1)]
        mm = multiclass_metrics(y[part], pred, proba, classes)
        metrics[f"{part}_macro_f1"] = mm["macro_f1"]
        metrics[f"{part}_weighted_f1"] = mm["weighted_f1"]
        metrics[f"{part}_log_loss"] = mm["log_loss"]
        for c_name, d in mm["per_class"].items():
            metrics[f"{part}_recall_{c_name}"] = d["recall"]
            metrics[f"{part}_precision_{c_name}"] = d["precision"]
            metrics[f"{part}_support_{c_name}"] = d["support"]
        if part == "test":
            (out / "per_class_metrics.json").write_text(json.dumps(mm, indent=2))
            save_figure(
                confusion_matrix_figure(mm["confusion"], classes), out / "confusion_matrix.png"
            )
    return metrics


def _explain_stage(
    cfg: TrainConfig, model: Any, X_val: np.ndarray, spec: FeatureSpec, out: Path
) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.shap_sample, len(X_val))
    sample = X_val[rng.choice(len(X_val), n, replace=False)]
    background = sample[: min(BACKGROUND_ROWS, n)]
    explainer = make_explainer(model, cfg.model, background)
    imp = global_importance(explainer, sample, spec.names)
    imp.to_csv(out / "shap_global_importance.csv", index=False)
    save_figure(shap_bar_figure(imp), out / "shap_summary.png")
    pd.DataFrame(background, columns=spec.names).to_parquet(
        out / "explainer_background.parquet", index=False
    )


def run_training(cfg: TrainConfig, settings: Settings) -> str:
    mlflow.set_tracking_uri(settings.resolved_tracking_uri())
    mlflow.set_experiment(cfg.experiment)
    spec = FEATURE_SPECS[cfg.feature_spec]
    parts = load_parts(
        settings.processed_dir, cfg.attempted_policy, SPLIT_COLUMN[cfg.split_strategy]
    )
    parts = {k: _select_rows(v, cfg.task) for k, v in parts.items()}
    X = {k: spec.to_matrix(v) for k, v in parts.items()}
    y = {k: _targets(v, cfg.task) for k, v in parts.items()}
    classes = list(s.FAMILY_CLASSIFIER_CLASSES) if cfg.task == "multiclass" else None
    if classes is not None:
        y_fit = {k: np.searchsorted(classes, v) for k, v in y.items()}
    else:
        y_fit = y

    with mlflow.start_run(run_name=cfg.run_name) as run, tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        mlflow.set_tags(
            {
                "dataset_variant": "distrinet_improved",
                "manifest_sha": manifest_digest(),
                "git_sha": _git_sha(),
                "split_strategy": str(cfg.split_strategy),
                "attempted_policy": str(cfg.attempted_policy),
                "feature_spec": cfg.feature_spec,
                "model": cfg.model,
                "weighting": cfg.weighting,
                "task": cfg.task,
            }
        )
        mlflow.log_params({"seed": cfg.seed, "max_fpr": cfg.max_fpr, "weighting": cfg.weighting})
        mlflow.log_params({f"p_{k}": v for k, v in cfg.params.items()})
        mlflow.log_params({f"rows_{k}": len(v) for k, v in parts.items()})

        model = build_model(
            cfg.model, cfg.task, cfg.params, cfg.seed, n_classes=len(classes) if classes else None
        )
        model = fit_model(
            model, X["train"], y_fit["train"], X["val"], y_fit["val"], cfg.weighting, cfg.model
        )

        if classes is None:
            metrics = _binary_stage(cfg, model, X, y, parts, out)
        else:
            metrics = _multiclass_stage(model, X, y, classes, out)

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(float(v))})
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        spec.save(out / "feature_spec.json")
        _explain_stage(cfg, model, X["val"], spec, out)

        mlflow.log_artifacts(str(out))
        _log_model(model, cfg.model, X["val"][:100])
        if cfg.register_as:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", cfg.register_as)
        headline = {
            k: round(float(v), 4)
            for k, v in metrics.items()
            if k.startswith("test_") and "support" not in k
        }
        log.info("run %s: %s", run.info.run_id, headline)
        return str(run.info.run_id)
```

- [ ] **Step 5: Implement `cli.py`**

```python
"""secops-train command line."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import typer

from secops.config import get_settings
from secops.detection.train import TrainConfig, run_training

app = typer.Typer(help="Train and report detection models.")

REPORT_COLS = [
    "run_name",
    "model",
    "weighting",
    "val_pr_auc",
    "test_pr_auc",
    "test_recall",
    "test_fpr",
    "test_precision",
    "test_f1",
    "test_roc_auc",
    "threshold",
    "val_macro_f1",
    "test_macro_f1",
]


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"expected key=value, got {item!r}")
        out[k] = v
    return out


@app.command()
def run(
    config: Path,
    set_: list[str] = typer.Option([], "--set", help="key=value overrides of the YAML config"),
) -> None:
    """Train one configuration and print the MLflow run id."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainConfig.from_yaml(config, _parse_overrides(set_))
    typer.echo(run_training(cfg, get_settings()))


@app.command()
def report(experiment: str) -> None:
    """Print a markdown table of all runs in an experiment (numbers come only from MLflow)."""
    mlflow.set_tracking_uri(get_settings().resolved_tracking_uri())
    df = mlflow.search_runs(experiment_names=[experiment])
    if df.empty:
        typer.echo("no runs")
        raise typer.Exit(code=1)
    rows = []
    for _, r in df.iterrows():
        row: dict[str, Any] = {
            "run_name": r.get("tags.mlflow.runName"),
            "model": r.get("tags.model"),
            "weighting": r.get("tags.weighting"),
            "run_id": str(r["run_id"])[:8],
        }
        for c in REPORT_COLS[3:]:
            v = r.get(f"metrics.{c}")
            row[c] = f"{v:.4f}" if isinstance(v, float) and pd.notna(v) else ""
        rows.append(row)
    typer.echo(pd.DataFrame(rows, columns=[*REPORT_COLS, "run_id"]).to_markdown(index=False))
```

- [ ] **Step 6: Run the integration tests**

Run: `uv run pytest tests/integration/test_train_pipeline.py -m integration -v`
Expected: 3 passed

- [ ] **Step 7: Run the six binary experiments on the full data**

```bash
export SECOPS_DATA_DIR=$HOME/data/secops
for m in logreg xgboost lightgbm; do
  uv run secops-train run --config configs/models/${m}_binary.yaml
  uv run secops-train run --config configs/models/${m}_binary.yaml --set weighting=none --set run_name=${m}_none
done
uv run secops-train report --experiment secops/detection-binary
```
Expected: six runs; the report prints a table. Record wall-clock per run in the commit message body.

- [ ] **Step 8: Commit**

```bash
git add src/secops/detection/train.py src/secops/detection/cli.py configs/models tests/integration/test_train_pipeline.py
git commit -m "feat(detection): MLflow training pipeline and secops-train CLI with report command

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Model registry helpers (champion alias, load with spec and threshold)

**Files:**
- Create: `src/secops/detection/registry.py`, `tests/unit/detection/test_registry.py`
- Modify: `src/secops/detection/cli.py` (add `promote-best`)

**Interfaces:**
- `DETECTOR_MODEL_NAME = "secops-detector"`, `FAMILY_MODEL_NAME = "secops-family-classifier"`, `CHAMPION_ALIAS = "champion"`; `promote(run_id, model_name, alias=CHAMPION_ALIAS) -> int` (registers the run's model, moves the alias, returns the version); `@dataclass LoadedModel(model, feature_spec, threshold, classes, model_name, version, run_id)` where `model.predict_proba(X)` returns shape `(n, n_classes)`; `load_model(model_name, alias=CHAMPION_ALIAS) -> LoadedModel`. Phase 2 depends on `load_model`.

- [ ] **Step 1: Write failing tests**

`tests/unit/detection/test_registry.py`:
```python
import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from secops.detection.features import FEATURE_SPEC_V1
from secops.detection.registry import CHAMPION_ALIAS, load_model, promote


@pytest.fixture
def tracking(tmp_path: Path) -> str:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test/registry")
    return uri


def _fake_run(workdir: Path) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    X = np.random.default_rng(0).normal(size=(50, 82)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    with mlflow.start_run() as run:
        FEATURE_SPEC_V1.save(workdir / "feature_spec.json")
        (workdir / "threshold.json").write_text(json.dumps({"threshold": 0.37, "method": "test"}))
        mlflow.log_artifact(str(workdir / "feature_spec.json"))
        mlflow.log_artifact(str(workdir / "threshold.json"))
        mlflow.sklearn.log_model(clf, "model", pyfunc_predict_fn="predict_proba")
        return str(run.info.run_id)


def test_promote_then_load_round_trip(tracking: str, tmp_path: Path) -> None:
    run_id = _fake_run(tmp_path / "a")
    assert promote(run_id, "test-detector", CHAMPION_ALIAS) == 1
    loaded = load_model("test-detector", CHAMPION_ALIAS)
    assert loaded.run_id == run_id
    assert loaded.threshold == pytest.approx(0.37)
    assert loaded.feature_spec == FEATURE_SPEC_V1
    assert loaded.classes is None
    proba = loaded.model.predict_proba(np.zeros((2, 82), dtype=np.float32))
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_promote_second_run_moves_alias(tracking: str, tmp_path: Path) -> None:
    r1 = _fake_run(tmp_path / "a")
    r2 = _fake_run(tmp_path / "b")
    promote(r1, "test-detector-2")
    assert promote(r2, "test-detector-2") == 2
    assert load_model("test-detector-2").run_id == r2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/detection/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'secops.detection.registry'`

- [ ] **Step 3: Implement `registry.py`**

```python
"""MLflow Model Registry helpers: promote a run to an alias, load a model with spec and threshold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
from mlflow import MlflowClient

from secops.detection.features import FeatureSpec

DETECTOR_MODEL_NAME = "secops-detector"
FAMILY_MODEL_NAME = "secops-family-classifier"
CHAMPION_ALIAS = "champion"


@dataclass
class LoadedModel:
    model: Any
    feature_spec: FeatureSpec
    threshold: float | None
    classes: list[str] | None
    model_name: str
    version: int
    run_id: str


class _ProbaAdapter:
    """Normalise pyfunc output to an (n, n_classes) probability matrix."""

    def __init__(self, pyfunc_model: Any) -> None:
        self._m = pyfunc_model

    def predict_proba(self, X: Any) -> np.ndarray:
        out = np.asarray(self._m.predict(X), dtype=float)
        if out.ndim == 1:  # positive-class probability only
            out = np.column_stack([1 - out, out])
        return out


def promote(run_id: str, model_name: str, alias: str = CHAMPION_ALIAS) -> int:
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.set_registered_model_alias(model_name, alias, mv.version)
    return int(mv.version)


def _artifact_json(run_id: str, name: str) -> dict[str, Any] | None:
    try:
        return dict(json.loads(mlflow.artifacts.load_text(f"runs:/{run_id}/{name}")))
    except Exception:  # noqa: BLE001  artifact absent for this task type
        return None


def load_model(model_name: str, alias: str = CHAMPION_ALIAS) -> LoadedModel:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    run_id = str(mv.run_id)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
    spec_json = _artifact_json(run_id, "feature_spec.json")
    if spec_json is None:
        raise ValueError(f"run {run_id} has no feature_spec.json artifact")
    thr = _artifact_json(run_id, "threshold.json")
    per_class = _artifact_json(run_id, "per_class_metrics.json")
    return LoadedModel(
        model=_ProbaAdapter(pyfunc),
        feature_spec=FeatureSpec.model_validate(spec_json),
        threshold=float(thr["threshold"]) if thr else None,
        classes=sorted(per_class["per_class"]) if per_class else None,
        model_name=model_name,
        version=int(mv.version),
        run_id=run_id,
    )
```

Implementer note: with the installed MLflow version, confirm what `pyfunc.predict` returns for each flavour logged in Task 10. The sklearn flavour needs `pyfunc_predict_fn="predict_proba"` (already set in `_log_model`). For the xgboost and lightgbm flavours, if `predict` returns class labels rather than probabilities, log those models through `mlflow.sklearn.log_model` with `pyfunc_predict_fn="predict_proba"` as well (both estimators are sklearn-compatible) and record the choice in the commit message. The test pins the contract: `predict_proba(X)` has shape `(n, n_classes)` and rows sum to 1.

- [ ] **Step 4: Add `promote-best` to `src/secops/detection/cli.py`**

```python
from secops.detection.registry import promote


@app.command(name="promote-best")
def promote_best(experiment: str, model_name: str, metric: str = "val_pr_auc") -> None:
    """Register the run with the highest validation metric and point the champion alias at it."""
    mlflow.set_tracking_uri(get_settings().resolved_tracking_uri())
    df = mlflow.search_runs(experiment_names=[experiment])
    col = f"metrics.{metric}"
    if df.empty or col not in df:
        typer.echo(f"no runs with {metric} in {experiment}")
        raise typer.Exit(code=1)
    best = df.sort_values(col, ascending=False).iloc[0]
    version = promote(str(best["run_id"]), model_name)
    typer.echo(f"{model_name} v{version} <- run {best['run_id']} ({metric}={best[col]:.4f})")
```

Tie rule from A13 (simpler model wins) is applied manually if two runs are within 0.0005 of each other; record the decision in `docs/evaluation.md`.

- [ ] **Step 5: Run tests, then promote the binary champion on the full-data runs**

Run: `uv run pytest tests/unit/detection/test_registry.py -v`
Expected: 2 passed

Run: `SECOPS_DATA_DIR=$HOME/data/secops uv run secops-train promote-best secops/detection-binary secops-detector`
Expected: prints the promoted version, run id and measured `val_pr_auc`.

- [ ] **Step 6: Commit**

```bash
git add src/secops/detection/registry.py src/secops/detection/cli.py tests/unit/detection/test_registry.py
git commit -m "feat(detection): model registry promote/load with feature spec and threshold

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Family classifier, held-out-day and ablation experiments

**Files:**
- Create: `configs/models/lightgbm_family.yaml`, `configs/models/xgboost_family.yaml`, `configs/models/lightgbm_binary_heldout.yaml`, `configs/models/ablation_withport.yaml`, `configs/models/ablation_drop.yaml`
- Modify: `Makefile` (add `train-binary`, `train-family`, `train-heldout`, `train-ablations`; make `train-all` call them in order)

No new library code. Verification is the existing integration test plus the logged runs.

- [ ] **Step 1: Write the configs**

`configs/models/lightgbm_family.yaml`:
```yaml
experiment: secops/detection-family
run_name: lightgbm_family_balanced
task: multiclass
model: lightgbm
weighting: balanced
feature_spec: v1-noport
split_strategy: chrono_within_group
attempted_policy: relabel_benign
shap_sample: 20000
```
`xgboost_family.yaml`: same with `model: xgboost`, `run_name: xgboost_family_balanced`.

`configs/models/lightgbm_binary_heldout.yaml`:
```yaml
experiment: secops/detection-heldout
run_name: lightgbm_balanced_heldout_friday
task: binary
model: lightgbm
weighting: balanced
feature_spec: v1-noport
split_strategy: heldout_friday
attempted_policy: relabel_benign
reuse_threshold_from_run: null   # supplied with --set at run time (chrono champion run id)
```

`configs/models/ablation_withport.yaml`: the champion binary config with `experiment: secops/detection-ablations`, `run_name: lightgbm_balanced_withport`, `feature_spec: v1-withport`.
`configs/models/ablation_drop.yaml`: the champion binary config with `experiment: secops/detection-ablations`, `run_name: lightgbm_balanced_attempted_drop`, `attempted_policy: drop`.

If the promoted binary champion is not LightGBM-balanced, change `model`/`weighting` in the held-out and ablation configs to match it before running: an ablation must change exactly one variable.

`Makefile` additions:
```makefile
train-binary:
	for m in logreg xgboost lightgbm; do \
	  uv run secops-train run --config configs/models/$${m}_binary.yaml; \
	  uv run secops-train run --config configs/models/$${m}_binary.yaml --set weighting=none --set run_name=$${m}_none; \
	done
	uv run secops-train promote-best secops/detection-binary secops-detector

train-family:
	for m in lightgbm xgboost; do \
	  uv run secops-train run --config configs/models/$${m}_family.yaml; \
	  uv run secops-train run --config configs/models/$${m}_family.yaml --set weighting=none --set run_name=$${m}_family_none; \
	done
	uv run secops-train promote-best secops/detection-family secops-family-classifier --metric val_macro_f1

train-heldout:
	uv run secops-train run --config configs/models/lightgbm_binary_heldout.yaml \
	  --set reuse_threshold_from_run=$$(uv run python -c "import mlflow;from mlflow import MlflowClient;from secops.config import get_settings as g;mlflow.set_tracking_uri(g().resolved_tracking_uri());print(MlflowClient().get_model_version_by_alias('secops-detector','champion').run_id)")

train-ablations:
	uv run secops-train run --config configs/models/ablation_withport.yaml
	uv run secops-train run --config configs/models/ablation_drop.yaml

train-all: train-binary train-family train-heldout train-ablations
```

- [ ] **Step 2: Run the experiments**

```bash
export SECOPS_DATA_DIR=$HOME/data/secops
make train-family train-heldout train-ablations
for e in secops/detection-family secops/detection-heldout secops/detection-ablations; do
  uv run secops-train report --experiment $e
done
```
Expected: 4 family runs, 1 held-out run, 2 ablation runs, all with artifacts; `secops-family-classifier@champion` set.

- [ ] **Step 3: Commit**

```bash
git add configs/models Makefile
git commit -m "feat(detection): family classifier, held-out-day and ablation experiment configs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Documentation and results (measured only)

**Files:**
- Create: `docs/dataset.md`, `docs/evaluation.md`, `docs/interview-notes.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/dataset.md`**

Sections, in order: source and citations (A1); three-layer distinction table; acquisition and verification with the hash table (A2); structure (A3); measured label distribution table (A3); family mapping and exclusions; data-quality findings (A4) with the post-build numbers pasted from `reports/relabel_benign/cleaning_report.json` (rows out, duplicates removed per label, inf cells); the Attempted policy with the authors' quoted directive; leakage table (A5); split strategy with per-split family counts from `split_report.json`; the rejected random split and why; known limitations (single testbed, residual label noise, profile-generated benign traffic, dedup shifts class balance).

- [ ] **Step 2: Write `docs/evaluation.md` (ML section)**

Paste the `secops-train report` output verbatim for each experiment under headings: binary baselines; family classifier plus the per-class table from `per_class_metrics.json` with `web_attack` flagged low-support; held-out Friday; ablations. Then "How to read these numbers": operating point definition, FPR budget, why PR-AUC, validation's double use, dedup caveat. Add the top-10 global SHAP features of the champion from `shap_global_importance.csv`. State the champion selection rule and the resulting registered versions. Include a footnote table mapping every pasted row to its MLflow run id.

- [ ] **Step 3: Write `docs/interview-notes.md` (Phase 1 entries)**

Concise answers (3–6 lines each): why PR-AUC not accuracy; why the FPR budget governs the threshold; why LightGBM/XGBoost over deep nets; how imbalance was handled and what the runs showed (cite run names); how leakage was prevented (ports, IPs, timestamps, duplicates, chronological split); why the corrected dataset and what "Attempted" means; what the held-out-day experiment shows; why MLflow; how retraining would work (registry alias swap); how drift would be detected (future: PSI on the top SHAP features). Any answer that depends on an unperformed run says `TBD`.

- [ ] **Step 4: Update `README.md`**

Keep the existing two-line description at the top. Add: setup (`uv`, `make setup`, `SECOPS_DATA_DIR`), dataset summary with the three-layer distinction and a link to `docs/dataset.md`, "Phase 1 results" table copied from `secops-train report` (binary headline rows and family headline row), the operating point, one sentence on the held-out-day result, and a "Status" section listing Phases 2–7 as not started. No number may appear that is not in MLflow.

- [ ] **Step 5: Verify, then commit**

Run: `make lint type test-unit test-integration`
Expected: all green.

Run: `grep -n "TBD" README.md docs/*.md`
Expected: only cells genuinely not measured.

```bash
git add README.md docs/dataset.md docs/evaluation.md docs/interview-notes.md
git commit -m "docs: dataset, evaluation results and interview notes for Phase 1

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Phase 1 completion check (A19)**

From a clean checkout of the branch, with `SECOPS_DATA_DIR` pointing at the data directory:
```bash
make setup && uv run secops-data verify && make data-build && make train-all
uv run python - <<'PY'
import mlflow, numpy as np, pandas as pd
from secops.config import get_settings
from secops.detection.metrics import binary_metrics
from secops.detection.registry import load_model
st = get_settings(); mlflow.set_tracking_uri(st.resolved_tracking_uri())
lm = load_model("secops-detector")
df = pd.read_parquet(st.processed_dir / "relabel_benign" / "flows.parquet")
test = df[df["split_chrono"] == "test"]
p = lm.model.predict_proba(lm.feature_spec.to_matrix(test))[:, 1]
m = binary_metrics(test["is_attack"].to_numpy(), p, lm.threshold)
logged = mlflow.get_run(lm.run_id).data.metrics["test_pr_auc"]
print("pr_auc reproduced:", m.pr_auc, "logged:", logged, "match:", abs(m.pr_auc - logged) < 1e-6)
PY
```
Expected: `match: True`. Record the outcome in the pull-request description.

---

## Self-review notes

- **Spec coverage:** A1–A19 map to Tasks 1–13. Deferred by design and documented as future work in Task 13: hyper-parameter search, cross-dataset generalisation, official-CSV ablation.
- **Placeholders:** the only `TBD`s are measured-value slots in Part A, which the user asked for explicitly. Part B has none; the held-out config's `reuse_threshold_from_run` is `null` in git and supplied at run time.
- **Type consistency:** `read_all(raw_dir, days=None, subdir="improved")`; `clean(df, policy) -> (df, CleaningReport)`; `assign_chrono_within_group(df, train_frac, val_frac, group_cols)`; `check_no_time_leak(df, split, group_cols)`; `build(raw_dir, processed_dir, reports_dir, policy, subdir, train_frac, val_frac)`; `FeatureSpec.to_matrix(df, strict_order=False)`; `build_model(name, task, params, seed, n_classes)`; `fit_model(model, X_train, y_train, X_val, y_val, weighting, name)`; `select_threshold(y, p, max_fpr) -> ThresholdChoice`; `make_explainer(model, name, X_background)`; `run_training(cfg, settings) -> str`; `promote(run_id, model_name, alias) -> int`; `load_model(model_name, alias) -> LoadedModel` are used with the same names and argument orders throughout.
- **Review Focus:** items 1–5 are pinned in Tasks 4, 5, 6, 7 and 8.
- **Known soft spot:** the pyfunc probability contract across MLflow flavours (Task 11 note). The registry test pins the behaviour so a wrong assumption fails at implementation time, not in Phase 2.
