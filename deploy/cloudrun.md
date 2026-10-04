# Deploying the detection API to Cloud Run

Everything below is run by a person with a Google Cloud project and `gcloud` installed (the
WSL environment used for development has no `gcloud`; in a Claude Code session, prefix a command
with `! ` to run it there). Nothing in this file claims the deployment exists until
`docs/deployment.md` carries the verification table produced by `scripts/verify_deployment.py`.

## What gets deployed

- The `cloudrun` image target: the Phase 2 runtime image plus the exported champion bundles
  copied into `/models` (Cloud Run has no persistent volume; a bundle is a few MB).
- The detection endpoints (`/health`, `/model`, `/predict`, `/predict/batch`) behind API keys
  held in Secret Manager, with the per-key rate limiter.
- The agent endpoints disabled (`SECOPS_INVESTIGATIONS_ENABLED=false`): the public deployment has
  no database and must not spend API money for strangers. The agent runs locally with Compose.

## Steps

```bash
# 0. once: login and pick a project / region
gcloud auth login
export PROJECT=<your-project-id> REGION=europe-west1

# 1. the API keys the service will accept (comma separated), stored in Secret Manager
printf '%s' "$(openssl rand -hex 24)" | gcloud secrets create secops-api-keys --data-file=- --project "$PROJECT"
gcloud secrets versions access latest --secret secops-api-keys --project "$PROJECT"   # keep it

# 2. export the champions into the build context
SECOPS_MODEL_DIR=deploy/models uv run secops-train export --model-name secops-detector --out deploy/models/detector
SECOPS_MODEL_DIR=deploy/models uv run secops-train export --model-name secops-family-classifier --out deploy/models/family

# 3. build with Cloud Build and deploy (idempotent; re-run to roll a new revision)
PROJECT=$PROJECT REGION=$REGION ./scripts/deploy_cloud_run.sh

# 4. verify and paste the table into docs/deployment.md
uv run python scripts/verify_deployment.py https://<service-url> <api-key>
```

Rollback: `gcloud run services update-traffic secops-api --region $REGION --to-revisions <previous>=100`.
Cost: scale-to-zero (`--min-instances 0`), 2 GiB / 1 vCPU, at most 2 instances; idle cost is zero,
a cold start loads the bundles in a few seconds (measured locally in `docs/api.md`).

## Why not the agent on Cloud Run

The agent needs the 1 GB event store, a database for investigations and an Anthropic key whose
spend is bounded by whoever can call the endpoint. Cloud SQL plus a Cloud Storage copy of the
store would make it work; that is a cost decision for an owner, not a portfolio default. The
Compose stack (`make compose-up`) runs the full platform on one machine.
