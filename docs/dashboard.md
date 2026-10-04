# Dashboard

A minimal Streamlit app (`dashboard/app.py`) that reads the API; the brief asks for three
views and nothing more.

| Page | Shows | Needs the API? |
|---|---|---|
| Overview | what the project is in four steps, the pipeline diagram, the headline table with a plain-language key, how to use the pages | no |
| Detector | the Phase 1 tables (binary detector, held-out Friday) read from the README, the PR-AUC figure, why the agent exists | no |
| Agent results | agent vs rule-based baseline, every metric explained, the four figures, every golden case with the verdict each configuration gave, three interview answers | no |
| Investigations | the four recorded Phase 4 scenarios and the 38 golden cases of the default configuration (ground truth shown, never given to the agent); live investigations and a box to start one when the API runs | no (live part: yes) |
| Investigation detail | one triage report end to end: the alert the detector raised, the tool calls in order, the summary, findings grouped as observed / model prediction / inference with their evidence ids, references, actions, uncertainties, raw JSON | no (live ids: yes) |
| Evaluation runs | every run file with its configuration and metrics, and an interactive regression-gate comparison between any two runs | no |

Everything offline comes from `evaluation/runs/*.json`, `evaluation/golden/v1.json`,
`tests/fixtures/llm/*/`, `docs/figures/` and the README tables (`dashboard/data.py`); nothing is
typed by hand. The sidebar says whether the API is reachable and how to start it.

From WSL (where the project's tooling and data live):

```bash
./scripts/dashboard.sh                         # Results page needs no API
SECOPS_API_KEY=dev-key ./scripts/dashboard.sh  # with `uv run secops-api` running on :8000
```

From Windows PowerShell (delegates to WSL; the browser URL is the same, http://localhost:8501):

```powershell
.\scripts\api.ps1 -ApiKey dev-key        # terminal 1, optional: the API
.\scripts\dashboard.ps1 -ApiKey dev-key  # terminal 2: the dashboard
```

Or `make dashboard`, or `docker compose --profile ui up --build`.

The API key comes from the environment, never from the page. The `dashboard` image target
installs Streamlit from the lockfile in the builder stage; the container was built and its
health endpoint answered 200 on 2026-10-04. Tests drive the app with
Streamlit's `AppTest` against a fake client (`tests/unit/dashboard/test_app.py`): empty state on
every page, a replayed report with its uncertainties, the runs table.
