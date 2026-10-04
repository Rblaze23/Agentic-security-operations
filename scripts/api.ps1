# Start the detection + agent API from Windows PowerShell by delegating to WSL.
#   .\scripts\api.ps1 -ApiKey dev-key
param([string]$ApiKey = "dev-key")
$repo = "/mnt/d/Internship/Agentic-security-operations"
wsl -e bash -lc "cd $repo && export PATH=`$HOME/.local/bin:`$PATH UV_PROJECT_ENVIRONMENT=`$HOME/.venvs/secops SECOPS_DATA_DIR=`$HOME/data/secops SECOPS_API_KEYS=$ApiKey MLFLOW_DISABLE_AGENT_HINT=1 && uv run secops-api"
