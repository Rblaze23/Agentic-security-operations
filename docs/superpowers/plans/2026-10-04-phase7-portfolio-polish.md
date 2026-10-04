# Phase 7 — Portfolio Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the finished platform legible in 60 seconds (a real ML model), 2 minutes (a tool-using agent) and 5 minutes (evaluation, reliability, security, MLOps, production engineering): a README with the brief's 16 sections, an architecture diagram, result charts generated from measured files, a minimal Streamlit dashboard, a demo walkthrough, and complete interview notes.

**Architecture:** Nothing new in the backend. The dashboard (`dashboard/app.py`, Streamlit) reads the API over HTTP with an API key (alert list and detail from `/investigations`, evaluation from `/evaluation/runs` plus the run files) and renders three pages: Alerts, Alert detail, Evaluation. Charts are produced by one script (`scripts/make_figures.py`) from `evaluation/runs/*.json`, `evaluation/golden/v1.json` and the Phase 1 report files, written to `docs/figures/*.png`, and referenced from the README; the script is re-runnable so figures never drift from the numbers. The architecture diagram is Mermaid in the README (rendered by GitHub) plus an exported SVG.

**Tech Stack:** Streamlit, matplotlib (already a dependency), Mermaid (README), the existing API.

**Spec:** brief §16 (frontend: alert list, alert detail, evaluation dashboard; "do NOT spend most of the project building UI"), §20 (README sections 1–16, never invent results), §24 (interview questions), §25 (60 s / 2 min / 5 min); design spec §10 row 7, §12.

## Global Constraints

- Work on `main`, uncommitted; never create branches, commits or pushes. Report the change set per task.
- Every number in the README and figures comes from a file in the repository (`evaluation/runs/*.json`, `evaluation/golden/v1.json`, `tests/fixtures/llm/*/result.json`, `docs/evaluation.md` Phase 1 tables) or is written `TBD`.
- No dashboard feature beyond the three pages the brief lists; no auth UI (the key is an environment variable); no new backend endpoints.
- Lint/type/test gates green at the end of every task.

## Review Focus

1. The dashboard must start and render all three pages against an API that has no investigations yet (empty lists, not tracebacks) → Task 2 `test_dashboard_renders_empty_state` (Streamlit `AppTest`).
2. The alert detail page must show tool errors and uncertainties as prominently as findings, so a report that escalated to a human reads as such → Task 2 `test_detail_page_shows_uncertainties`.
3. `scripts/make_figures.py` must fail loudly when a run file is missing rather than drawing an empty chart → Task 1 `test_make_figures_requires_inputs`.
4. The README must not contain the strings "TODO", "lorem" or any metric that is not in a source file: a test greps every number in the "Results" section against the run files → Task 3 `test_readme_numbers_come_from_runs`.
5. The demo walkthrough must be runnable in replay mode without an API key, so a reviewer can follow it offline → Task 4 `scripts/demo.sh --replay` exercised in CI.

---

### Task 1: Figures from measured files

**Files:** `scripts/make_figures.py`, `docs/figures/` (generated PNGs, committed), `tests/unit/test_make_figures.py`

Figures: (a) `agent_vs_baseline.png`: grouped bars for verdict accuracy, evidence recall, grounding, severity-within-one, composite, per investigator, from `evaluation/runs/agent-v1-k1.json` and `baseline-rule-based.json`; (b) `verdict_by_kind.png`: verdict accuracy by case kind (attack / benign_fp / adversarial) for both; (c) `cost_latency.png`: cost per case and latency p50/p95; (d) `reliability.png`: per-case verdict agreement across k=3 repeats from `agent-v1-k3.json`; (e) `detector_pr.png`: PR-AUC per model from the Phase 1 table in `docs/evaluation.md` (parsed, not retyped). Each figure carries the run id and date in its caption.

- [ ] Test: run the script on the real files → all PNGs exist and are non-trivial (> 10 KB); run with a missing file → `FileNotFoundError` naming it.
- [ ] Implement with matplotlib (no seaborn), `Agg` backend; dpi 150.

### Task 2: Streamlit dashboard (minimal)

**Files:** `dashboard/__init__.py`, `dashboard/app.py`, `dashboard/client.py` (HTTP wrapper with API key from `SECOPS_API_KEYS` / `SECOPS_API_URL`), `tests/unit/dashboard/test_app.py`, `pyproject.toml` (optional group `dashboard = ["streamlit>=1.38"]`, script `secops-dashboard`), `docker-compose.yml` (`dashboard` service, profile `ui`), `Dockerfile` (target `dashboard`), `docs/dashboard.md`.

Pages:
- **Alerts**: table from `GET /investigations?limit=100`: severity (colour), family, verdict, confidence, created_at, status; a text input to paste an `Alert` JSON and `POST /investigations`.
- **Alert detail**: `GET /investigations/{id}`: the original alert (metadata, detector probability, top SHAP features), investigation steps (tool, arguments, evidence id, status), findings grouped by kind with evidence ids, techniques and CVEs with their evidence, uncertainties and critic issues, recommended actions, cost and models.
- **Evaluation**: `GET /evaluation/runs` table; the figures from Task 1; the latest `compare` table rendered from the two newest run files.

- [ ] Tests with `streamlit.testing.v1.AppTest` and a `respx`/`responses`-style fake of the API (use `httpx.MockTransport`): empty state, detail page with the `ftp_bruteforce` report, evaluation page lists runs.

### Task 3: README with the 16 sections and the diagram

**Files:** `README.md` (restructured), `docs/architecture.md` (Mermaid source + `docs/figures/architecture.svg` exported with `mmdc` if available, else the Mermaid block only), `tests/unit/test_readme.py`.

Sections in the brief's order: 1 Project overview (60-second paragraph with the headline numbers), 2 Problem statement, 3 Why agentic architecture (with the measured agent-vs-baseline sentence), 4 Architecture diagram (Mermaid: data → detector → API → alert → LangGraph graph with tools → PostgreSQL/Langfuse → dashboard), 5 ML pipeline, 6 Agent workflow, 7 Tool architecture, 8 Evaluation methodology, 9 Results (three tables: detection, agent vs baseline, reliability; figures), 10 Security considerations, 11 MLOps, 12 Local setup, 13 API documentation, 14 Deployment (status line from `docs/deployment.md`), 15 Limitations, 16 Future work. Keep the roadmap table at the end. Every existing section moves rather than being rewritten; the numbers stay as they are.

- [ ] `tests/unit/test_readme.py`: the 16 headings exist in order; no "TODO"/"TBD" outside the Deployment and Future work sections; every `$x.xxx` and `0.xxx` in Results appears in a run file or `docs/evaluation.md`.

### Task 4: Demo walkthrough, interview notes, future work

**Files:** `docs/demo.md`, `scripts/demo.sh` (replay mode: start the API with `SECOPS_AGENT_MODE=replay`, POST the recorded `ftp_bruteforce` alert, GET the report, print the summary; `--live` variant), `docs/interview-notes.md` (every brief §24 question answered or cross-referenced; add retraining/drift and scaling answers), `docs/future-work.md`, `.github/workflows/ci.yml` (`scripts/demo.sh --replay` step).

- [ ] Interview notes: check the 18 questions in brief §24 against the existing headings; write the missing ones (how the agent decides which tool to call; how to scale; retraining and drift via MLflow alias swap and feature-distribution PSI; what happens at low confidence) with references to measured numbers.
- [ ] Future work: Cloud SQL + GCS-backed agent deployment, shared rate-limit store and investigation queue, critic precision tuning measured against the golden set, per-flow payload rendering for `predict_attack`, PostgreSQL single-pass aggregate for DoS bursts, model retraining and drift monitoring, Langfuse verification.

### Task 5: Final whole-repository review and closing report

- [ ] Dispatch the reviewer on the Phase 5–7 range (base `1699d20`), fix Critical/Important findings test-first, update the ledgers, run the full gates, and write the closing summary for the user with the complete change set and the three commands they must run themselves (commit, Cloud Run deploy, optional Langfuse keys).

## Self-review

- Spec coverage: brief §16 three pages → Task 2; §20 sections 1–16 and "never invent results" → Task 3 with a test; §24 → Task 4; §25 timing → Task 3 ordering (overview, diagram and results first); design spec §10 row 7 (diagrams, benchmark tables, charts, demo walkthrough, interview notes) → Tasks 1–4.
- Deviation stated: the dashboard reads the API, not the database directly (brief: "reading from the API", design A3), and is an optional dependency group so the core install stays small.
- Review Focus: 1–2 → Task 2; 3 → Task 1; 4 → Task 3; 5 → Task 4.
