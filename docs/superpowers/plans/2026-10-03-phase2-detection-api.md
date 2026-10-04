# Phase 2 — Detection API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the Phase 1 champions behind a typed FastAPI service that turns a flow record into a structured prediction and, above the operating threshold, an alert carrying the top SHAP contributions, from a self-contained model bundle that runs identically on a laptop, in Docker Compose, and later on Cloud Run.

**Architecture:** Training keeps using the MLflow registry; serving never touches MLflow. A new `secops-train export` command materialises a registry alias into a **model bundle** directory (estimator, feature spec, threshold, classes, explainer background, provenance). The API loads bundles from `SECOPS_MODEL_DIR` at startup into one `DetectorService` (binary detector + family classifier + SHAP explainer) and exposes `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /model`. All request and response shapes are Pydantic v2 models in a new `secops.schemas` package that Phases 3–5 reuse (the `Alert` and `Prediction` schemas are the agent's input). API-key authentication is a FastAPI dependency; rate limiting, persistence and tracing stay in Phase 6.

**Tech Stack:** FastAPI, uvicorn, Pydantic v2 (+ pydantic-settings), httpx (tests), Docker multi-stage with uv, Docker Compose, GitHub Actions `docker build` job. No database, no queue, no Redis.

**Spec:** `docs/superpowers/specs/2026-10-03-platform-architecture-design.md` sections 3.3 (services), 5 (schemas), 9 (security); project brief sections 5 (model serving), 6 (alert model), 13 (security). Phase 1 interfaces consumed: `secops.detection.registry.load_model`, `secops.detection.features.FeatureSpec`, `secops.detection.explain.make_explainer` / `top_k_contributions`, `secops.detection.metrics`, champions `secops-detector@champion` and `secops-family-classifier@champion`.

## Global Constraints

- Python 3.12, uv, `uv.lock` committed; venv via `UV_PROJECT_ENVIRONMENT` as in Phase 1.
- No model files, bundles, data or secrets in git. Bundles live under `SECOPS_MODEL_DIR` (default `$SECOPS_DATA_DIR/models`), git-ignored.
- Identifier fields (IPs, ports, timestamps) travel as **metadata** in requests and never enter the feature matrix; the feature matrix is built only through `FeatureSpec.to_matrix` from the bundle's spec.
- The operating threshold is the bundle's `threshold.json`; the API never recomputes or overrides it.
- Every endpoint validates input with Pydantic (bounded batch size, exact feature-name set, finite floats or null). Unknown fields are rejected.
- All secrets (API keys) come from environment variables. Containers run as non-root with a health check.
- Never fabricate: `GET /model` reports the bundle's recorded run id, model version and metrics as exported; nothing is typed by hand.
- Commit after every task with the attribution line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. **A request whose `features` object is missing one of the 82 names, has an extra name, or spells one with a different case** must be rejected with 422 naming the offending keys, never silently zero-filled. → Task 1 `test_flow_features_rejects_missing_and_unknown_names`.
2. **A feature value of `null`** is accepted and becomes NaN for the trees (the rate features are NaN after cleaning); `NaN`/`Infinity` literals are not valid JSON and must produce 422, not a 500. → Task 1 `test_flow_features_null_becomes_nan`, Task 4 `test_predict_rejects_non_json_numbers`.
3. **A bundle whose feature spec version differs from the one the family classifier was trained with**, or whose `classes.json` order does not match the estimator's `classes_`, must fail at startup (`/health` reports not ready and the process exits non-zero), never serve mislabelled families. → Task 2 `test_bundle_rejects_mismatched_feature_spec`, Task 3 `test_detector_service_validates_class_order`.
4. **A batch of 1,001 items** is rejected with 422 before any inference; a batch of 1,000 identical rows returns 1,000 predictions with identical scores and distinct `prediction_id`s. → Task 4 `test_batch_limit_and_ids`.
5. **A request without an API key, or with a wrong one**, gets 401 from `/predict` and `/predict/batch` while `/health` stays open; keys are compared in constant time. → Task 4 `test_auth_required_on_predict_only`.

---

# Part A — Design

## A1. Why a bundle, not MLflow at serving time

MLflow's registry is the right source of truth for *which* model is the champion, but a serving container that needs a tracking database, an artifact store and the MLflow client to answer `/predict` is slower to start, harder to secure and impossible to run on Cloud Run without a shared database. A bundle directory is a plain filesystem contract: `export` resolves the alias once at build time and writes everything inference needs; the API has no MLflow dependency at runtime. Promotion of a new champion is `export` + redeploy, which Phase 6 automates. Alternatives considered: MLflow model serving (`mlflow models serve`) gives no custom schemas, no SHAP, no alert logic; loading `models:/name@champion` at startup couples the service to the registry's availability.

Bundle layout (`$SECOPS_MODEL_DIR/<name>/`):

```
detector/
  MLmodel + model.pkl … (mlflow.sklearn artefact directory, cloudpickle)
  feature_spec.json         # names + version, from the run
  threshold.json            # operating point, from the run
  explainer_background.parquet
  bundle.json               # {model_name, version, run_id, alias, exported_at, metrics: {...}, git_sha, manifest_sha}
family/
  MLmodel + …  feature_spec.json  classes.json  bundle.json
```

`bundle.json` copies the run's logged test metrics and tags verbatim so `GET /model` reports measured numbers.

## A2. Request and response contract

```
POST /predict            PredictRequest  -> PredictResponse
POST /predict/batch      BatchPredictRequest {items: list[PredictRequest] (1..1000)} -> BatchPredictResponse
GET  /health             HealthResponse  {status: ok|starting|error, detector_loaded, family_loaded, bundle_versions}
GET  /model              ModelInfo       {detector: BundleInfo, family: BundleInfo, feature_spec_version, threshold}
```

`PredictRequest`:
- `event_id: str | None` (caller's correlation id, ≤ 128 chars)
- `metadata: FlowMetadata` — `timestamp: datetime | None`, `source_ip: IPvAnyAddress | None`, `destination_ip`, `source_port: int 0..65535 | None`, `destination_port`, `protocol: int 0..255 | None`
- `features: dict[str, float | None]` — exactly the 82 names of the bundle's feature spec

`PredictResponse` embeds `Prediction`:
- `prediction_id: str` (UUID4), `event_id`, `model_name`, `model_version`, `run_id`, `feature_spec_version`
- `attack_probability: float`, `threshold: float`, `is_alert: bool`
- `predicted_family: str | None`, `family_probabilities: dict[str, float] | None` (only when `is_alert`)
- `top_contributions: list[FeatureContribution]` (k = 5, SHAP for the positive class)
- `timestamp: datetime` (UTC, server time), `latency_ms: float`

`Alert` (in `secops.schemas.alert`, produced by the API when `is_alert` and consumed by Phase 4): `alert_id`, `event_id`, `metadata`, `prediction`, `key_features: dict[str, float]` (the five top-contribution features and their values), `status: Literal["new"]`. `PredictResponse.alert` is `Alert | None`.

Errors: 422 (validation, Pydantic detail), 401 (missing or wrong API key), 503 (bundle not loaded), 500 never leaks internals (generic message plus request id in the log).

## A3. Service layout

```
src/secops/
  schemas/      __init__.py  flow.py  prediction.py  alert.py  api.py
  detection/    bundle.py (export + load)   [+ cli.py: export command]
  api/          __init__.py  settings.py  auth.py  detector.py  app.py  routes.py  logging.py
```

`DetectorService` (in `api/detector.py`) is a plain class with `predict(rows: list[PredictRequest]) -> list[Prediction]`; it holds the two estimators, the spec, the threshold, the class list and a SHAP explainer built once from the detector bundle's background sample. It is constructed in the FastAPI lifespan and stored on `app.state`; routes depend on it through a `get_service()` dependency, so tests inject a service built from a tiny fixture bundle.

Family inference runs only for rows above threshold (saves work and keeps non-alerts free of a family label that would be meaningless).

## A4. Security in this phase

- `X-API-Key` header checked by a dependency against `SECOPS_API_KEYS` (comma-separated); `hmac.compare_digest`; `/health` is exempt; `/model` and `/docs` require the key (the docs reveal the feature schema; harmless, but consistent).
- Pydantic `extra="forbid"` on every request model; batch capped at 1,000; string fields length-capped; `features` keys must equal the spec set.
- Container: `python:3.12-slim`, uv-installed from `uv.lock`, non-root user, read-only bundle mount, `HEALTHCHECK` on `/health`.
- Deferred to Phase 6 (documented, not forgotten): rate limiting, TLS termination, request signing, persistence of predictions, Langfuse/OTel tracing.

## A5. Testing strategy

- **Unit, schemas**: exact-name validation, null→NaN, bounds, forbid-extra, Alert construction.
- **Unit, bundle**: `export` from a temporary MLflow run (reusing the Phase 1 integration fixture path) produces the layout above; `load_bundle` round-trips; mismatched spec / class order fail loudly.
- **Unit, service**: a session-scoped fixture trains a tiny LightGBM detector + family classifier on the 986-row fixture (fast, < 5 s) and exports a bundle to `tmp_path`; tests check threshold logic, family only on alerts, top-5 ordering, batch equivalence to single calls.
- **API**: `httpx.AsyncClient` against the app with the fixture service injected: all endpoints, auth, 422 paths, batch limits, health before/after load, `/model` content equals `bundle.json`.
- **Container**: CI job builds the image and runs `GET /health` against it with a fixture bundle baked in as a build arg; no real champion in CI.
- **Manual on full champions**: `secops-train export` from the real registry, `docker compose up api`, `curl` a flow from the test split and confirm the probability equals the Phase 1 completion-check value for that row.

## A6. Documentation

`docs/api.md` (contract, auth, examples, error table, bundle format), README section "Serving", interview notes (bundle vs registry at serve time; why family only above threshold; API key as dependency; why no rate limiting yet; what the SHAP contributions are for).

## A7. Milestones

| # | Deliverable | Tasks |
|---|---|---|
| M1 | `secops.schemas` with tests | 1 |
| M2 | bundle export/load + `secops-train export` | 2 |
| M3 | `DetectorService` on fixture bundle | 3 |
| M4 | FastAPI app, auth, endpoints, tests | 4 |
| M5 | Dockerfile, Compose, CI build job, docs | 5 |

Definition of done: `make test` green including API tests; `secops-train export` on the real champions then `docker compose up api` answers `/health` ok and `/predict` for a test-split row with the probability logged in Phase 1; `docs/api.md` and README updated; no bundle or key in git.

## A8. Risks

| Risk | Mitigation |
|---|---|
| SHAP TreeExplainer on every request adds latency | explainer built once; batch SHAP in one call; measure p50/p95 in the API tests and record in docs; k=5 only |
| cloudpickle bundle is a code-execution boundary | bundle dir is read-only, built from the same `uv.lock`, provenance in `bundle.json`; stated in docs/security |
| Feature names with `/` and spaces in JSON | they are plain JSON object keys; validated as an exact set; no aliasing |
| Fixture-trained models in tests give degenerate thresholds | tests assert behaviour relative to the bundle's threshold, not fixed numbers |
| DrvFs mount drops mid-task | commit per task; data and models live on the Linux filesystem |

---

# Part B — Tasks

Branch `phase-2/detection-api` from `main`. Each task: tests first, run to see them fail, implement, run green, lint and type-check, commit.

### Task 1: Schemas package

**Files:** `src/secops/schemas/{__init__,flow,prediction,alert,api}.py`, `tests/unit/schemas/{__init__,test_flow,test_prediction,test_api}.py`

**Interfaces produced:**
- `FlowMetadata(timestamp, source_ip, destination_ip, source_port, destination_port, protocol)` all optional.
- `FlowFeatures` helper: `validate_features(features: dict[str, float | None], spec: FeatureSpec) -> dict[str, float]` raising `ValueError` listing `missing=[...] unknown=[...]`; null → `math.nan`.
- `PredictRequest(event_id, metadata, features)` with `extra="forbid"`; `BatchPredictRequest(items: list[PredictRequest], min 1, max 1000)`.
- `FeatureContribution(feature, value, shap_value)` (Pydantic twin of the explain dataclass).
- `Prediction(...)` as in A2; `Alert(...)`; `PredictResponse(prediction, alert)`, `BatchPredictResponse(predictions, alerts)`; `HealthResponse`; `BundleInfo`; `ModelInfo`.

Key tests (write first): exact-name validation (missing/unknown/case), null→NaN, `extra="forbid"`, batch bounds, IP/port validation, `Alert.from_prediction(request, prediction)` picks `key_features` from `top_contributions`.

### Task 2: Model bundle export and load

**Files:** `src/secops/detection/bundle.py`, `src/secops/detection/cli.py` (add `export`), `tests/unit/detection/test_bundle.py`, `.gitignore` (`models/`)

**Interfaces produced:**
- `export_bundle(model_name: str, out_dir: Path, alias="champion") -> BundleManifest` — resolves the alias via `MlflowClient`, downloads `models:/<name>@<alias>` with `mlflow.artifacts.download_artifacts` into `out_dir/model/`, copies `feature_spec.json`, `threshold.json` (if present), `classes.json` (if present), `explainer_background.parquet` from the source run, writes `bundle.json` with name, version, run id, alias, UTC timestamp, the run's tags (`git_sha`, `manifest_sha`, `feature_spec`, `model`, `weighting`) and its logged metrics.
- `load_bundle(dir: Path) -> LoadedBundle(estimator, feature_spec, threshold, classes, background: np.ndarray | None, manifest: BundleManifest)` using `mlflow.sklearn.load_model(dir / "model")` (returns the native estimator, which SHAP needs; no tracking server involved). Validates: `feature_spec.names` length equals the estimator's `n_features_in_`; when `classes.json` exists, `list(estimator.classes_)` indexes match its length and order is recorded.
- CLI: `secops-train export --model-name secops-detector --out $SECOPS_MODEL_DIR/detector [--alias champion]`.

Tests: export from a temporary MLflow run created by `run_training` on the fixture (reuse the Phase 1 integration fixture), load, predict equals the run's logged probabilities on 5 rows; mismatched spec raises; missing `threshold.json` for a family bundle is fine.

### Task 3: DetectorService

**Files:** `src/secops/api/__init__.py`, `src/secops/api/detector.py`, `tests/unit/api/{__init__,conftest,test_detector}.py`

**Interfaces produced:**
- `DetectorService.from_dirs(detector_dir, family_dir | None, k=5)`; `.predict(requests: list[PredictRequest]) -> list[Prediction]`; `.info() -> ModelInfo`; `.ready: bool`.
- Behaviour: builds the matrix through `feature_spec.to_matrix` on a DataFrame assembled from validated feature dicts; positive-class probability from `predict_proba`; `is_alert = p >= threshold`; family `predict_proba` only for alert rows; SHAP via `make_explainer(estimator, model_kind, background)` once, `top_k_contributions` per row (positive class); latency measured per call.

Fixture (`tests/unit/api/conftest.py`, session scope): train LightGBM binary + family on the fixture build with `n_estimators=30`, export both bundles to a temp dir, construct the service. Tests: threshold boundary (a row at exactly the threshold is an alert), family only on alerts, top-5 sorted by |shap|, batch == per-row results, class order validation failure.

### Task 4: FastAPI application

**Files:** `src/secops/api/{settings,auth,logging,routes,app}.py`, `tests/unit/api/test_app.py`, `pyproject.toml` (fastapi, uvicorn[standard], httpx dev)

**Interfaces produced:**
- `ApiSettings(BaseSettings, env_prefix="SECOPS_")`: `model_dir: Path`, `api_keys: list[str]` (parsed from comma-separated env), `top_k: int = 5`, `batch_limit: int = 1000`, `log_level`.
- `require_api_key` dependency (constant-time compare; 401 with `WWW-Authenticate: ApiKey`).
- `create_app(service: DetectorService | None = None, settings: ApiSettings | None = None) -> FastAPI` with lifespan loading bundles when no service is injected; routes `/health`, `/model`, `/predict`, `/predict/batch`; JSON logging with request id and latency; generic 500 handler.
- `secops-api` console script running uvicorn (`src/secops/api/__main__.py`).

Tests with `httpx.AsyncClient(transport=ASGITransport(app))`: health before load (`starting`, 503 on predict) and after; `/model` equals bundle manifests; predict happy path and alert object presence; 422 on bad features / extra fields / batch 1001; 401 without key, 200 with key; `/health` open; non-JSON numbers rejected; latency fields present. p50/p95 over 200 single requests printed and recorded in `docs/api.md`.

### Task 5: Container, Compose, CI, docs

**Files:** `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `Makefile` (targets `export-models`, `api`, `docker-build`, `compose-up`), `.github/workflows/ci.yml` (add `docker` job), `docs/api.md`, `README.md`, `docs/interview-notes.md`, `.env.example` (`SECOPS_API_KEYS`, `SECOPS_MODEL_DIR`)

- Dockerfile: builder stage `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` → `uv sync --frozen --no-dev`; runtime stage `python:3.12-slim` with `libgomp1`, non-root `app` user, copies venv + `src`, `HEALTHCHECK CMD curl -f http://localhost:8000/health`, `CMD ["secops-api"]`.
- Compose: `api` service with `${SECOPS_MODEL_DIR}` mounted read-only at `/models`, env from `.env`, port 8000.
- CI: `docker build` job plus a smoke test using a fixture bundle produced in the job by the Task 3 fixture code (`uv run python -m tests.fixtures.make_bundle`).
- Manual verification recorded in `docs/api.md`: export real champions, compose up, predict one test-split row, compare the probability with Phase 1's completion check.

---

## Self-review notes

- Spec coverage: brief §5 endpoints and structured prediction (A2, Task 4); §6 alert schema with metadata fields (Task 1); §13 input validation, auth, secrets by env, no shell access (A4); Phase 0 §3.3 single service, no extra infrastructure. Rate limiting and persistence are explicitly deferred to Phase 6 per the spec's phase table.
- Open decision for the user: API-key auth in Phase 2 (recommended, small) versus deferring all auth to Phase 6.
- Placeholders: none; tests are named in each task and will be written in full at execution time following the Phase 1 pattern.
- Types: `Prediction` and `Alert` live in `secops.schemas` from this phase on; Phase 1's `FeatureContribution` dataclass gets a Pydantic twin rather than being replaced, to keep the training code dependency-free of the API layer.
