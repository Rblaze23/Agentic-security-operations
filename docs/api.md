# Detection API

The Phase 2 service turns one flow record into a typed prediction and, above the operating
threshold, an alert. It serves the Phase 1 champions from a **model bundle** directory and has no
MLflow dependency at runtime.

## Running it

```bash
export SECOPS_DATA_DIR=$HOME/data/secops
make export-models                 # registry alias -> $SECOPS_MODEL_DIR/{detector,family}
export SECOPS_API_KEYS=dev-key     # or put it in .env
make api                           # uvicorn on :8000 (SECOPS_PORT)
curl -s localhost:8000/health
```

Container: `make docker-build` then `make compose-up` (mounts `$SECOPS_MODEL_DIR` read-only at
`/models`, reads keys from `.env`). Locally `secops-api` binds `127.0.0.1` (`SECOPS_HOST`); the
image sets `SECOPS_HOST=0.0.0.0` so the port can be published. `make docker-smoke` builds fixture bundles, starts the image,
and checks `/health` and `/predict`; CI runs the same script.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | `ok` when both bundles are loaded, `degraded` when only the detector is (no family labels; a warning is logged at startup), `starting` before the lifespan ran |
| GET | `/model` | key | bundle provenance: model name, registry version, run id, alias, export time, feature-spec version, model kind, the run's logged metrics and tags, operating threshold, `top_k` |
| POST | `/predict` | key | one flow → `{prediction, alert | null}` |
| POST | `/predict/batch` | key | up to 1,000 flows → `{predictions, alerts}` |
| GET | `/docs`, `/openapi.json` | key | Swagger UI and schema |

Authentication: header `X-API-Key`, compared in constant time (on bytes, so non-ASCII input is
simply wrong rather than an error) against the comma-separated `SECOPS_API_KEYS`. An empty key
list refuses every protected route (fail closed) and logs an error at startup. The check runs in
middleware **before the request body is read**, and again as a route dependency; failures return
401 with `WWW-Authenticate: ApiKey realm="secops"`. Bodies whose `Content-Length` exceeds
`SECOPS_MAX_BODY_BYTES` (default 8 MiB) are refused with 413 before being read.

### Request

```json
{
  "event_id": "flow-0001",
  "metadata": {
    "timestamp": "2017-07-04T12:19:20.152476Z",
    "source_ip": "172.16.0.1", "destination_ip": "192.168.10.50",
    "source_port": 52108, "destination_port": 21, "protocol": 6
  },
  "features": { "Protocol": 6, "Flow Duration": 4.0, "...": "all 82 names from feature_spec.json", "Total TCP Flow Time": 4.0 }
}
```

Rules enforced by the schema (`src/secops/schemas/flow.py`):

- `features` must contain exactly the 82 names of the bundle's feature spec, spelled exactly;
  missing or unknown names give 422 listing both sets. Values are floats or `null`; `null` becomes
  NaN, which the tree models handle natively. `NaN` and `Infinity` are not JSON and give 422.
- `metadata` fields are optional, validated (IP addresses, 0–65535 ports, 0–255 protocol) and are
  **never used as features**; they travel through to the alert.
- Unknown top-level fields are rejected; `event_id` is capped at 128 characters; a batch holds
  1 to 1,000 items.

### Response

```json
{
  "prediction": {
    "prediction_id": "6f1c…", "event_id": "flow-0001",
    "model_name": "secops-detector", "model_version": 1, "run_id": "dcc36c43…",
    "feature_spec_version": "v1-noport",
    "attack_probability": 0.9993, "threshold": 0.000242, "is_alert": true,
    "predicted_family": "brute_force",
    "family_probabilities": {"botnet": 0.0, "brute_force": 0.99, "ddos": 0.0, "dos": 0.01, "port_scan": 0.0, "web_attack": 0.0},
    "top_contributions": [{"feature": "Bwd Packet Length Std", "value": 0.0, "shap_value": -2.1}, "… 5 items, sorted by |shap|"],
    "timestamp": "2026-10-03T19:05:12.345678Z", "latency_ms": 7.4
  },
  "alert": {
    "alert_id": "9a2e…", "event_id": "flow-0001", "status": "new",
    "metadata": {"…": "as sent"},
    "prediction": {"…": "same object"},
    "key_features": {"Bwd Packet Length Std": 0.0, "…": "the five top-contribution features and their values"}
  }
}
```

- `is_alert` is `attack_probability >= threshold`, the threshold being the bundle's
  `threshold.json` (chosen on validation in Phase 1; the API never re-tunes it).
- The family classifier runs only for alert rows; non-alerts carry `predicted_family: null`.
- `top_contributions` are SHAP values for the positive class, computed once per batch; a `value`
  of `null` means the feature was NaN.
- `latency_ms` is the wall time of the `DetectorService.predict` call the row belonged to (the whole
  batch for batch requests).
- Every response carries `X-Request-ID`; the same id is in the JSON request log line.

### Errors

| Status | When |
|---|---|
| 401 | missing or wrong `X-API-Key` (or no keys configured); decided before the body is parsed |
| 413 | declared body larger than `SECOPS_MAX_BODY_BYTES` |
| 422 | schema violation, feature-name drift (`item <i>: feature set mismatch: missing=[…] unknown=[…]`, at most ten names each), batch outside 1..1000, non-finite numbers. Schema errors list `type`, `loc`, `msg` and never echo input values |
| 503 | bundles not loaded (only possible before the lifespan ran) |
| 500 | any other error, including internal `ValueError`s from the estimators; body is `{"detail": "internal error", "request_id": …}`, the traceback is logged, and the response still carries `X-Request-ID` |

## Model bundles

`secops-train export --model-name <registered name> --out <dir> [--alias champion]` resolves the
alias in the MLflow registry and writes:

```
<dir>/
  model/                        mlflow.sklearn artefact (cloudpickle), loaded with mlflow.sklearn.load_model
  feature_spec.json             ordered feature names + version (from the run)
  threshold.json                operating point (binary bundles)
  classes.json                  family order used at training time (family bundles)
  explainer_background.parquet  1,000-row background sample for SHAP
  bundle.json                   provenance: name, version, run_id, alias, exported_at, feature_spec_version,
                                model_kind, the run's metrics and tags (git_sha, manifest_sha, …)
```

Export stages into a temporary directory, makes it world-readable (0755 directories, 0644 files,
because the container runs as a non-root user and `mkdtemp` would otherwise leave 0700), then
swaps it in by renaming the previous bundle to `<dir>.old`, renaming the new one into place and
deleting the backup; a failure at any point leaves the previous bundle where it was. `load_bundle` checks that the estimator's `n_features_in_` matches the spec
and that `classes.json` has one entry per estimator class; the *order* of the classes is taken
from the file, which `export` copies from the training run, so a hand-edited bundle could mislabel
families (documented limitation). The service additionally refuses a family bundle whose feature
spec differs from the detector's, a detector bundle without a threshold, and a family bundle
without classes. Any of these failures aborts startup; uvicorn exits non-zero.

Security note: the bundle is cloudpickle, so the directory it is loaded from is a code-execution
trust boundary. It is mounted read-only, produced by the same `uv.lock` the image is built from,
and carries its provenance; Phase 6 moves it to a bucket with the same guarantees.

## Measured (2026-10-03/04)

**In-process** (`TestClient`, fixture bundle, one flow per request, SHAP included): p50 8.0 ms,
p95 9.3 ms over 100 calls (`tests/unit/api/test_app.py::test_single_request_latency`). A
1,000-row batch answers in about 100 ms.

**Container, fixture bundles** (`scripts/docker_smoke.py` against `secops-api:local`, image
1.61 GB, Docker Desktop 29.6 on Windows with the build context on `D:`): image build 389 s cold;
container healthy 6 s after start; `/health` ok, `/predict` valid, bad payload 422; whole smoke
run 34 s including the fixture-bundle build.

**Container, real champions** (bundles exported from `secops-detector@champion` and
`secops-family-classifier@champion`, copied to a `D:` path and bind-mounted read-only, because
Docker Desktop's WSL integration is off on this machine and the engine cannot mount the WSL
filesystem; a Windows bind mount does not enforce POSIX modes, which is why the 0700 bundle
permission defect was only caught in review and is now pinned by
`test_exported_bundle_is_readable_by_other_users`): `/model` reports run `dcc36c43`,
version 1, threshold 0.000242, logged `test_pr_auc` 0.99993. Seven test-split rows scored in the
container matched the in-process `DetectorService` probabilities exactly:

| Row label | Probability | Alert | Family | HTTP round trip | `latency_ms` inside |
|---|---|---|---|---|---|
| DoS Hulk | 0.999965 | yes | dos | 26 ms | 18.6 |
| DDoS | 0.999965 | yes | ddos | 19 ms | 12.1 |
| Portscan | 0.999621 | yes | port_scan | 20 ms | 12.6 |
| FTP-Patator | 0.999938 | yes | brute_force | 18 ms | 12.7 |
| Botnet | 0.999861 | yes | botnet | 21 ms | 15.0 |
| Web Attack - XSS | 0.402631 | yes | web_attack | 23 ms | 14.4 |
| BENIGN | 0.000010 | no | — | 15 ms | 10.3 |

The first three SSH-Patator rows of the test split scored 0.000127–0.000155, below the threshold,
in both the container and in-process: those are among the 4.5% of SSH-Patator flows the champion
misses (`docs/evaluation.md` §2.1), reproduced faithfully rather than hidden.

## Phase 6 additions (2026-10-04)

**Rate limiting.** Every protected route runs a per-API-key token bucket
(`SECOPS_RATE_LIMIT_PER_MINUTE`, default 60; `0` disables). When a key has no token left the
response is `429` with a `Retry-After` header (seconds). The bucket lives in the API process;
behind more than one replica a shared store (Redis) would be needed, which is noted, not built.
`/health` is never limited.

**Investigations (the agent over HTTP).**

| Endpoint | What it does |
|---|---|
| `POST /investigations` (body `{"alert": Alert}`) | Starts the LangGraph investigation as a background task and returns `202` with `{investigation_id, alert_id, status: "queued"}`. A second request for an alert that is still running returns the same id instead of paying twice. |
| `GET /investigations/{id}` | Status (`queued`, `running`, `done`, `failed`) plus the full `TriageReport`, verdict, severity and cost when done; a failed run carries the error class and message. |
| `GET /investigations?limit=20` | Recent investigations from the database plus the ones still in flight. |
| `GET /evaluation/runs` | One summary per `evaluation/runs/*.json` (composite, verdict accuracy, grounding rate, cost). |

`SECOPS_INVESTIGATIONS_ENABLED=false` turns the agent endpoints into `503` (the public Cloud
Run deployment runs this way: no persistent database, no API spend from the internet).
`SECOPS_AGENT_MODE=replay` with `SECOPS_AGENT_FIXTURE_ROOT` and `SECOPS_AGENT_SCENARIO` serves a
recorded scenario, which is how the endpoint is tested without a model.

**PostgreSQL.** `SECOPS_DATABASE_URL=postgresql+psycopg://…` switches the event store and the
investigations to PostgreSQL (migration 0003 also creates `model_predictions`, schema only:
no writer yet); Compose
starts one. Migrations are applied by the agent CLI and the API on first use (idempotent).

Still open after Phase 6: TLS is terminated by Cloud Run or a reverse proxy, never by the app;
bundle distribution from object storage; a shared rate-limit store; a queue for investigations
across replicas.
