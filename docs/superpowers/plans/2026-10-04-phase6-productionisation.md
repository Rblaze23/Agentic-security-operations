# Phase 6 — Productionisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the whole platform (detection API, investigation agent, persistence, evaluation) as one deployable service with PostgreSQL, tracing, authentication and rate limiting, security checks in CI, Docker Compose locally, and a real Cloud Run deployment.

**Architecture:** One FastAPI process (`secops.api`) keeps the Phase 2 detection endpoints and gains the agent endpoints: `POST /investigations` starts an investigation as a background task (idempotent per alert), `GET /investigations/{id}` and `GET /investigations` read the Phase 4 repository, `GET /evaluation/runs` reads `evaluation/runs/*.json`. The database is the Phase 3/4 SQLAlchemy + Alembic schema, now proven on PostgreSQL (Compose service, CI service container) with SQLite kept for tests. Tracing is a thin `secops.observability` layer that records one span per LLM call and per tool call to Langfuse when `LANGFUSE_*` keys are set and stays silent otherwise. Compose runs `api` + `postgres`; Cloud Run runs the same image with Cloud SQL or, for the portfolio deployment, an ephemeral SQLite with the limitation documented.

**Tech Stack:** FastAPI, SQLAlchemy 2 + Alembic + `psycopg[binary]`, `langfuse` SDK (optional dependency group), Docker multi-stage image (exists), Docker Compose, GitHub Actions (`services: postgres`), `bandit`, `pip-audit`, `gitleaks`, Google Cloud Run + Artifact Registry (`gcloud` run by the user).

**Spec:** `docs/superpowers/specs/2026-10-03-platform-architecture-design.md` §3.3 (services), §3.4 (technology reasons), §9 (security posture), §10 row 6, §11 (risks). `docs/api.md` "What Phase 6 adds". `docs/security.md` "Explicitly not covered yet".

## Global Constraints

- Work on `main`, uncommitted; never create branches, commits or pushes. Report the change set per task.
- No secrets in git: `.env` is ignored; Compose reads `env_file: .env`; Cloud Run secrets come from Secret Manager or `--set-env-vars` typed by the user. `gitleaks` runs in CI.
- Nothing in docs claims to be deployed until it is: the Cloud Run section carries the real service URL and the `curl` output the user pastes back, or `TBD`.
- API calls that cost money (the agent endpoint) are behind the API key and the rate limiter; the integration test for the endpoint uses `mode=replay` on the recorded `ftp_bruteforce` scenario, never live.
- Lint/type/test gates green at the end of every task: `uv run ruff check . && uv run mypy && uv run pytest tests/unit -p no:warnings`.
- Docker is available only through `/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe` with `D:\` paths; `gcloud` is not installed in WSL and must be run by the user (interactive login), so Task 6 produces the exact commands and a verification script and stops.

## Review Focus

1. A second `POST /investigations` for the same `alert_id` while the first is still running must not start a second paid investigation; it returns the existing id with status `running` → Task 3 `test_post_investigation_is_idempotent_per_alert`.
2. An investigation that raises inside the background task must leave a `failed` row with the error, not a row stuck in `running` forever → Task 3 `test_background_failure_is_recorded`.
3. Rate limiting must be per API key, not global, and must return `429` with `Retry-After` → Task 2 `test_rate_limit_is_per_key_and_sets_retry_after`.
4. The PostgreSQL migration chain must be applied from zero on a fresh database and be idempotent, with the event store's indexes and the investigation tables present → Task 1 `test_postgres_migrations_from_zero` (integration, `postgres` marker).
5. Tracing must never break an investigation: a Langfuse outage or a bad key is logged once and the run completes with the same report → Task 4 `test_tracing_failure_is_swallowed`.

---

## File structure

```
src/secops/db/session.py             make_engine: PostgreSQL pool settings, read-only role note
src/secops/db/migrations/versions/0003_predictions.py   model_predictions table (API predictions persisted)
src/secops/api/settings.py           +database_url passthrough, rate_limit_per_minute, investigations_enabled
src/secops/api/ratelimit.py          TokenBucket per key, middleware/dependency
src/secops/api/investigations.py     routes: POST/GET /investigations, GET /evaluation/runs
src/secops/api/app.py                wires the new router, startup migrations (opt-in), repository
src/secops/observability/__init__.py Tracer protocol, NullTracer, LangfuseTracer, get_tracer()
src/secops/agent/llm.py              tracer hook per call (span: model, tokens, cost, latency)
src/secops/agent/tools.py            tracer hook per tool call
docker-compose.yml                   api + postgres (+ healthchecks, volumes); dashboard in Phase 7
deploy/cloudrun.md                   step-by-step with the exact gcloud commands
scripts/deploy_cloud_run.sh          builds, pushes, deploys (run by the user)
scripts/verify_deployment.py         hits /health, /model, /predict, /investigations on a URL
.github/workflows/ci.yml             +postgres service job, bandit, pip-audit, gitleaks
tests/integration/test_postgres.py   migrations + repository on the CI postgres (marker postgres)
tests/unit/api/test_ratelimit.py, tests/unit/api/test_investigations.py, tests/unit/observability/test_tracer.py
docs/deployment.md, docs/security.md (Phase 6 section), docs/api.md (new endpoints), README
```

---

### Task 1: PostgreSQL everywhere

**Files:**
- Modify: `src/secops/db/session.py` (engine options for PostgreSQL: `pool_pre_ping=True`, `pool_size=5`; keep the SQLite pragma), `pyproject.toml` (`psycopg[binary]>=3.2`), `docker-compose.yml` (postgres service), `.env.example` (`SECOPS_DATABASE_URL=postgresql+psycopg://secops:secops@localhost:5432/secops` commented), `.github/workflows/ci.yml` (job `postgres` with `services: postgres:16`, runs `pytest tests/integration -m postgres`)
- Create: `src/secops/db/migrations/versions/0003_predictions.py` (table `model_predictions(prediction_id PK, event_id, alert_id nullable, attack_probability, threshold, is_alert, predicted_family, model_name, model_version, created_at)` + index on `created_at`), `tests/integration/test_postgres.py`
- Test: `tests/unit/db/test_migration.py` (+0003 columns on SQLite)

**Interfaces:**
- Produces: `make_engine(url, read_only=False)` unchanged signature; `secops.db.models.ModelPrediction`; `upgrade_to_head(url)` works for `postgresql+psycopg://` URLs.

- [ ] Write the failing unit test for 0003 (columns and index on SQLite) and the integration test:

```python
# tests/integration/test_postgres.py
import os
import pytest
from sqlalchemy import inspect, text
from secops.db.session import make_engine, upgrade_to_head

pytestmark = pytest.mark.postgres
URL = os.environ.get("SECOPS_TEST_DATABASE_URL", "postgresql+psycopg://secops:secops@localhost:5432/secops_test")


def test_postgres_migrations_from_zero() -> None:
    upgrade_to_head(URL)
    upgrade_to_head(URL)  # idempotent
    insp = inspect(make_engine(URL))
    assert {"events", "load_runs", "investigations", "tool_calls", "model_predictions"} <= set(insp.get_table_names())
    assert ("source_ip", "ts_us") in {tuple(i["column_names"]) for i in insp.get_indexes("events")}


def test_repository_round_trip_on_postgres(...) -> None:  # reuse tests/unit/agent/test_repository._run with the fixture registry
    ...
```

- [ ] Add the `postgres` marker to `pyproject.toml` (excluded by default like `network` and `llm`), implement the migration and model, run the SQLite unit tests, then run the integration test against a local PostgreSQL started with Docker (`docker.exe run -d --name secops-pg -e POSTGRES_USER=secops -e POSTGRES_PASSWORD=secops -e POSTGRES_DB=secops_test -p 5432:5432 postgres:16`) and record the result in the ledger; if Docker is not reachable, state it and rely on the CI job.
- [ ] Lint, type-check, change set.

---

### Task 2: API keys with per-key rate limiting

**Files:**
- Create: `src/secops/api/ratelimit.py`, `tests/unit/api/test_ratelimit.py`
- Modify: `src/secops/api/settings.py` (`rate_limit_per_minute: int = 60`), `src/secops/api/app.py` (dependency on protected routes), `docs/api.md` (429 row)

**Interfaces:**

```python
class TokenBucket:  # per key; capacity = rate, refill rate/60 per second; monotonic clock injectable
    def __init__(self, rate_per_minute: int, now: Callable[[], float] = time.monotonic) -> None
    def take(self, key: str) -> tuple[bool, float]   # (allowed, seconds until a token is available)

def rate_limited(request: Request) -> None   # FastAPI dependency: 429 + Retry-After when exhausted; keyed by the API key
```

- [ ] Tests: two keys with rate 2/min: key A exhausts after 2 calls (third → 429 with `Retry-After` ≥ 1), key B still allowed; advancing the injected clock by 30 s restores one token; `/health` is never limited.
- [ ] Implement; wire the dependency into the protected routes (`/predict*`, `/model`, `/investigations*`); document.

---

### Task 3: Investigation endpoints

**Files:**
- Create: `src/secops/api/investigations.py`, `tests/unit/api/test_investigations.py`
- Modify: `src/secops/api/app.py`, `src/secops/api/settings.py` (`investigations_enabled: bool = True`, `agent_mode: Literal["live","replay"] = "live"`, `agent_fixture_root: Path | None`), `src/secops/schemas/api.py` (`InvestigationRequest{alert: Alert}`, `InvestigationStatus{investigation_id, alert_id, status: queued|running|done|failed, verdict?, severity?, cost_usd?, error?}`), `docs/api.md`

**Interfaces:**

```python
POST /investigations  (API key, rate limited) body InvestigationRequest -> 202 InvestigationStatus(status="queued")
GET  /investigations/{investigation_id} -> InvestigationStatus + report (TriageReport) when done
GET  /investigations?limit=20 -> list[InvestigationStatus]
GET  /evaluation/runs -> list of {run_id, finished_at, cases, composite, cost_total_usd} from evaluation/runs/*.json
```

Behaviour: an in-process `InvestigationManager` keeps `running: dict[alert_id, investigation_id]`; a second POST for a running alert returns the same id (202). The background task builds deps once per app (`_build` with the detector), runs `run_investigation`, saves through `InvestigationRepository`, and on exception writes a row with `status="failed"` and the error in `uncertainties` (report minimal). Tests use `agent_mode=replay` with `tests/fixtures/llm` and the `ftp_bruteforce` alert: POST → 202, GET until `done`, report verdict `true_positive`; idempotency; failure recorded (monkeypatch `run_investigation` to raise).

- [ ] Tests first (FastAPI `TestClient`, background tasks run on exit of the request context), implement, document.

---

### Task 4: Tracing (Langfuse, optional)

**Files:**
- Create: `src/secops/observability/__init__.py`, `tests/unit/observability/__init__.py`, `tests/unit/observability/test_tracer.py`
- Modify: `src/secops/agent/llm.py` (`LLM(..., tracer: Tracer | None = None)`; after each call `tracer.llm_call(model, request_id, usage, latency_ms, cost_usd)`), `src/secops/agent/tools.py` (`ToolExecutor(..., tracer)`; `tracer.tool_call(name, status, latency_ms, evidence_id)`), `src/secops/agent/graph.py` (`AgentDeps.tracer`; `tracer.start(investigation_id, alert_id)` / `tracer.end(report summary)`), `pyproject.toml` (optional group `tracing = ["langfuse>=2"]`), `.env.example` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`), `docs/agent.md`

**Interfaces:**

```python
class Tracer(Protocol):
    def start(self, investigation_id: str, alert_id: str, prompt_version: str) -> None: ...
    def llm_call(self, model: str, request_id: str | None, usage: Usage, latency_ms: float, cost_usd: float) -> None: ...
    def tool_call(self, name: str, evidence_id: str, status: str, latency_ms: float) -> None: ...
    def end(self, verdict: str, severity: str, cost_usd: float) -> None: ...

class NullTracer: ...      # default
class RecordingTracer: ... # for tests: lists of events
class LangfuseTracer: ...  # wraps langfuse.Langfuse; every method is try/except with one warning
def get_tracer(settings: AgentSettings) -> Tracer   # Langfuse when keys set and the package imports, else Null
```

- [ ] Tests: `RecordingTracer` sees start → n llm_call → m tool_call → end with the totals equal to `UsageTotals`; a tracer whose methods raise does not change the report (`test_tracing_failure_is_swallowed`: the graph wraps tracer calls); `get_tracer` returns `NullTracer` without keys.
- [ ] Implement; verify against Langfuse cloud only if the user provides keys (otherwise document as `TBD: not exercised against a Langfuse instance`).

---

### Task 5: Compose, CI security checks, Docker smoke

**Files:**
- Modify: `docker-compose.yml` (api depends_on postgres healthy; `SECOPS_DATABASE_URL`; volume for `events.db` optional; `secops-data load-events` as a one-off `tools` profile), `.github/workflows/ci.yml` (+`security` job: `uv run bandit -r src -q`, `uv run pip-audit`, `gitleaks/gitleaks-action@v2`), `pyproject.toml` (dev deps bandit, pip-audit), `Makefile` (compose-up uses postgres), `scripts/docker_smoke.py` (also checks `GET /investigations` returns 200 with the key)
- [ ] Run bandit and pip-audit locally and fix or document findings; build the image with `docker.exe` and run the smoke script; record measured image size and startup time in `docs/deployment.md`.

---

### Task 6: Cloud Run deployment (user runs gcloud)

**Files:**
- Create: `deploy/cloudrun.md`, `scripts/deploy_cloud_run.sh`, `scripts/verify_deployment.py`, `docs/deployment.md`
- Modify: `README.md` (Deployment section, Phase 6 row), `docs/security.md` (Phase 6 section: edge TLS by Cloud Run, secrets in Secret Manager, non-root image, rate limits, what is still open)

Steps the user runs (each printed with `! ` so it runs in this session): `gcloud auth login`, `gcloud config set project <id>`, `gcloud services enable run.googleapis.com artifactregistry.googleapis.com`, `gcloud artifacts repositories create secops --repository-format=docker --location=<region>`, `gcloud builds submit --tag <region>-docker.pkg.dev/<project>/secops/api:<sha>` (Cloud Build avoids the WSL Docker path problem), `gcloud run deploy secops-api --image … --region … --allow-unauthenticated --set-secrets SECOPS_API_KEYS=secops-api-keys:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest --set-env-vars SECOPS_MODEL_DIR=/models,SECOPS_INVESTIGATIONS_ENABLED=false --memory 2Gi --cpu 1 --min-instances 0`. Model bundles: baked into a deploy-time image layer (`Dockerfile` target `cloudrun` that `COPY`s `deploy/models/` built by `make export-models`), since Cloud Run has no persistent volume; the agent endpoints stay disabled on the public deployment unless Cloud SQL is provisioned (documented decision: the portfolio deploy proves the detection service; the agent runs locally with Compose).

- [ ] `scripts/verify_deployment.py <url> <key>`: `/health` 200, `/model` shows the champion, `/predict` on a fixture row returns a probability, `/investigations` → 404 or 503 when disabled; prints a table for the docs.
- [ ] Docs: `docs/deployment.md` with the measured verification output pasted by the user (or `TBD`), cost notes (scale-to-zero), rollback (`gcloud run services update-traffic`).

---

## Self-review

- Spec coverage: §3.3 services (api + postgres in Compose; mlflow dev-only; langfuse via cloud keys; dashboard deferred to Phase 7 as decided in Phase 0) → Tasks 1, 4, 5. §9 security (API key + per-key rate limiting, secrets via env, bandit/pip-audit/gitleaks, non-root, health checks, threat model) → Tasks 2, 5, 6. §10 row 6 done-when ("CI green; service reachable on Cloud Run with health check") → Task 6 verification script. §11 risks: cost (agent endpoints behind key + limiter; disabled on the public deploy).
- Deviation stated: Langfuse is wired through an optional tracer and verified only if keys are available; the self-hosted Langfuse stack (ClickHouse, Redis, MinIO) is not added to Compose because it would triple the Compose footprint for a dev-only concern; the Phase 0 decision "no dashboard until Phase 7" stands.
- Review Focus: 1 → Task 3; 2 → Task 3; 3 → Task 2; 4 → Task 1; 5 → Task 4.
