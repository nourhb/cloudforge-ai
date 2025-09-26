# CloudForge AI Demo Script (Windows PowerShell)
# Prereqs: Local stack running (AI:5001, Backend:4000, Frontend:3002)

Write-Host "== CloudForge AI Demo ==" -ForegroundColor Cyan

# 1) IaC generation
Write-Host "[1/4] IaC: Generating Service YAML..." -ForegroundColor Yellow
$body = '{"prompt":"Expose backend as ClusterIP on port 4000"}'
try {
  $iac = Invoke-RestMethod -UseBasicParsing -Uri http://localhost:4000/api/iac/generate -Method Post -ContentType application/json -Body $body
  Write-Host "IaC ok:" -ForegroundColor Green
  $iac | ConvertTo-Json -Depth 10
} catch { Write-Host "IaC failed: $($_.Exception.Message)" -ForegroundColor Red }

# 2) Marketplace upload + list
Write-Host "[2/4] Marketplace: Uploading small worker file..." -ForegroundColor Yellow
$tmp = Join-Path $pwd 'demo-upload.txt'
Set-Content -Path $tmp -Value 'hello from cloudforge demo'
$curl = "$env:SystemRoot\System32\curl.exe"
& $curl -s -X POST "http://localhost:4000/api/marketplace/upload" -F "name=echo-api" -F "runtime=python:3.12" -F "file=@$tmp"
Write-Host "\nListing marketplace items:" -ForegroundColor Yellow
& $curl -s "http://localhost:4000/api/marketplace/list" | Write-Output
Remove-Item $tmp -Force

# 3) OTP demo
Write-Host "[3/4] OTP: requesting and verifying (demo) ..." -ForegroundColor Yellow
try {
  $req = Invoke-RestMethod -UseBasicParsing -Uri http://localhost:4000/api/auth/request-otp -Method Post -ContentType application/json -Body '{"identifier":"you@example.com"}'
  if ($req.ok -and $req.code) {
    $verify = Invoke-RestMethod -UseBasicParsing -Uri http://localhost:4000/api/auth/verify -Method Post -ContentType application/json -Body ("{`"identifier`":`"you@example.com`",`"code`":`"$($req.code)`"}")
    Write-Host "OTP verify result:" -ForegroundColor Green
    $verify | ConvertTo-Json -Depth 10
  } else { Write-Host "OTP request did not return a code (expected in demo)." -ForegroundColor Yellow }
} catch { Write-Host "OTP flow failed: $($_.Exception.Message)" -ForegroundColor Red }

# 4) Health & Metrics quick check
Write-Host "[4/4] Health & Metrics" -ForegroundColor Yellow
try { Invoke-WebRequest -UseBasicParsing http://localhost:4000/health | Select-Object -ExpandProperty Content | Write-Output } catch {}
try { Invoke-WebRequest -UseBasicParsing http://localhost:5001/health | Select-Object -ExpandProperty Content | Write-Output } catch {}
try { Invoke-WebRequest -UseBasicParsing http://localhost:4000/metrics | Select-Object -ExpandProperty Content | Write-Output } catch {}
try { Invoke-WebRequest -UseBasicParsing http://localhost:5001/metrics | Select-Object -ExpandProperty Content | Write-Output } catch {}

Write-Host "== Demo complete ==" -ForegroundColor Cyan
