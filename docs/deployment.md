# Deployment

## Local: Docker Compose (api + postgres)

```bash
cp .env.example .env            # set SECOPS_API_KEYS and ANTHROPIC_API_KEY
make export-models              # champions -> $SECOPS_MODEL_DIR/{detector,family}
make compose-up                 # postgres (healthcheck) then api, migrations applied on first use
curl -s -H "X-API-Key: $KEY" localhost:8000/health
```

The API container mounts the bundles read-only and `$SECOPS_DATA_DIR` read-only (event store,
ATT&CK index, NVD cache); investigations are persisted in the `postgres` service
(`SECOPS_DATABASE_URL` is set by Compose). `make compose-down` keeps the `pgdata` volume.

Measured on 2026-10-04 (WSL2, Docker Desktop):

| What | Result |
|---|---|
| Runtime image size (`secops-api:local`, after Phase 6 dependencies) | 1.69 GB (1,688,778,525 bytes; Phase 2 measured 1.61 GB before psycopg and the agent stack) |
| PostgreSQL migrations 0001–0003 from zero (postgres:16 container) | applied and idempotent; integration tests pass |
| Docker smoke (`scripts/docker_smoke.py`, fixture bundles, container on port 18000) | /health ok, /predict 27.9 ms in-container |
| Dashboard image (`--target dashboard`, Streamlit from the lockfile) | container starts, `/_stcore/health` 200 |

## Cloud Run

Procedure and reasoning: `deploy/cloudrun.md`. Script: `scripts/deploy_cloud_run.sh`.
Verification: `scripts/verify_deployment.py`, whose output is pasted below once a deployment
exists.

**Status: TBD — no Cloud Run deployment has been performed from this repository yet.** The
steps are ready; running them needs a Google Cloud project and an interactive `gcloud auth
login`, which the maintainer performs.

<!-- VERIFICATION TABLE -->

## Security checks in CI

`bandit -r src -ll` (no medium or high findings; two justified `nosec`s: the bundle directory
mode and a non-security SHA-1 cache key), `pip-audit` (no known vulnerabilities on
2026-10-04), `gitleaks` on the full history, plus the PostgreSQL integration job and the
recorded smoke evaluation.
