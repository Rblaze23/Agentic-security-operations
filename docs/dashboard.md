# Dashboard

A minimal Streamlit app (`dashboard/app.py`) that reads the API; the brief asks for three
views and nothing more.

| Page | Shows | Source |
|---|---|---|
| Alerts | severity, verdict, status, cost, timestamp per investigation; a box to paste an `Alert` JSON and start an investigation | `GET /investigations`, `POST /investigations` |
| Alert detail | the original alert and detector output, the tool-call timeline, findings by kind with evidence ids, ATT&CK / CVE references, uncertainties and critic notes, recommended actions, the raw report | `GET /investigations/{id}` |
| Evaluation | the evaluation runs table, the figures generated from the run files, the latest before/after comparison with its gates | `GET /evaluation/runs`, `evaluation/runs/*.json`, `docs/figures/` |

```bash
uv sync --group dashboard
export SECOPS_API_URL=http://localhost:8000 SECOPS_API_KEY=<key>
make dashboard                      # or: docker compose --profile ui up --build
```

The API key comes from the environment, never from the page. The `dashboard` image target
installs Streamlit from the lockfile in the builder stage; the container was built and its
health endpoint answered 200 on 2026-10-04. Tests drive the app with
Streamlit's `AppTest` against a fake client (`tests/unit/dashboard/test_app.py`): empty state on
every page, a replayed report with its uncertainties, the runs table.
