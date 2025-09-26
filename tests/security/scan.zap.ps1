# OWASP ZAP Baseline Scan (Windows PowerShell)
# Requires Docker Desktop running
# Scans frontend (3002) and backend (4000) for baseline issues

param(
  [string]$FrontendUrl = "http://host.docker.internal:3002",
  [string]$BackendUrl = "http://host.docker.internal:4000"
)

Write-Host "Starting ZAP Baseline scan against $FrontendUrl and $BackendUrl" -ForegroundColor Cyan

$WorkDir = (Get-Location).Path

# Frontend scan
docker run --rm -t -v "${WorkDir}:/zap/wrk" owasp/zap2docker-stable zap-baseline.py `
    -t "$FrontendUrl" `
    -r zap-frontend-report.html `
    -m 5

# Backend scan
docker run --rm -t -v "${WorkDir}:/zap/wrk" owasp/zap2docker-stable zap-baseline.py `
    -t "$BackendUrl" `
    -r zap-backend-report.html `
    -m 5

Write-Host "Reports saved as zap-frontend-report.html and zap-backend-report.html" -ForegroundColor Green
