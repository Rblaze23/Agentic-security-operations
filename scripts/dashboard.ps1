# Start the dashboard from Windows PowerShell by delegating to WSL, where the project runs.
#   .\scripts\dashboard.ps1              # Results page works without the API
#   .\scripts\dashboard.ps1 -ApiKey dev-key   # with the API running (see scripts\api.ps1)
param([string]$ApiKey = "")
$repo = "/mnt/d/Internship/Agentic-security-operations"
$env:WSLENV = "SECOPS_API_KEY/u"
$env:SECOPS_API_KEY = $ApiKey
wsl -e bash -lc "cd $repo && ./scripts/dashboard.sh"
