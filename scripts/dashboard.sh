#!/usr/bin/env bash
# Start the Streamlit dashboard from WSL. Run from the repository root:
#   ./scripts/dashboard.sh                 # Results page works without the API
#   SECOPS_API_KEY=dev-key ./scripts/dashboard.sh   # with a running API on :8000
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/secops}"
export SECOPS_DATA_DIR="${SECOPS_DATA_DIR:-$HOME/data/secops}"
export SECOPS_API_URL="${SECOPS_API_URL:-http://localhost:8000}"
export MLFLOW_DISABLE_AGENT_HINT=1
uv sync --group dashboard >/dev/null
echo "dashboard: http://localhost:8501  (Ctrl+C to stop)"
exec uv run streamlit run dashboard/app.py --server.headless true --server.port 8501
