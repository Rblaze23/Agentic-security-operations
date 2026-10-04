#!/usr/bin/env bash
# Build the image with Cloud Build and deploy it to Cloud Run. Run by a human with gcloud:
#   PROJECT=my-project REGION=europe-west1 ./scripts/deploy_cloud_run.sh
# Prerequisites (once): gcloud auth login; the two secrets created in Secret Manager:
#   printf '%s' "<api keys, comma separated>" | gcloud secrets create secops-api-keys --data-file=-
# The image bakes the exported bundles from deploy/models (make export-models first).
set -euo pipefail
: "${PROJECT:?set PROJECT}"; : "${REGION:?set REGION}"
SERVICE=${SERVICE:-secops-api}
TAG=${TAG:-$(git rev-parse --short HEAD)}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/secops/api:${TAG}"

gcloud services enable --project "$PROJECT" run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
gcloud artifacts repositories describe secops --project "$PROJECT" --location="$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create secops --project "$PROJECT" --repository-format=docker --location="$REGION"
test -d deploy/models/detector || { echo "deploy/models/detector missing: run 'make export-models' with SECOPS_MODEL_DIR=deploy/models"; exit 1; }

gcloud builds submit --project "$PROJECT" --tag "$IMAGE" --timeout=20m .
gcloud run deploy "$SERVICE" --project "$PROJECT" \
  --image "$IMAGE" --region "$REGION" --platform managed --allow-unauthenticated \
  --port 8000 --memory 2Gi --cpu 1 --min-instances 0 --max-instances 2 --concurrency 8 \
  --set-env-vars "SECOPS_MODEL_DIR=/models,SECOPS_INVESTIGATIONS_ENABLED=false,SECOPS_RATE_LIMIT_PER_MINUTE=60,SECOPS_HOST=0.0.0.0" \
  --set-secrets "SECOPS_API_KEYS=secops-api-keys:latest"
URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')
echo "deployed: $URL"
echo "verify with: uv run python scripts/verify_deployment.py $URL <one of the api keys>"
