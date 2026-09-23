# One-command local demo: Ollama (+ model warm-up), optional Redis, backend, frontend.
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -SkipWarmup -NoRedis
# Press any key in this window to stop the backend and frontend it started
# (or pass -NoWait to leave them running and get the stop command printed).
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [string[]]$WarmupModels = @("llama3.2", "gemma3:4b"),
    [switch]$SkipWarmup,
    [switch]$NoOllama,
    [switch]$NoRedis,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"
$rootDir = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $rootDir "frontend"
$backendDir = Join-Path $rootDir "backend"
$logDir = Join-Path $rootDir "logs"
$null = New-Item -ItemType Directory -Path $logDir -Force
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backendLog = Join-Path $logDir "demo-backend-$timestamp.log"
$frontendLog = Join-Path $logDir "demo-frontend-$timestamp.log"
$ollamaUrl = "http://127.0.0.1:11434"
$backendUrl = "http://127.0.0.1:$BackendPort"

function Write-Step    { param([string]$M) Write-Host "  >> $M" -ForegroundColor Yellow }
function Write-Success { param([string]$M) Write-Host "  OK $M" -ForegroundColor Green }
function Write-Warn    { param([string]$M) Write-Host "  !! $M" -ForegroundColor Red }
function Write-Info    { param([string]$M) Write-Host "     $M" -ForegroundColor White }

# PowerShell 5.1 with ErrorActionPreference=Stop turns a native command's stderr into a
# terminating error; run native tools through this and rely on $LASTEXITCODE instead.
function Invoke-Native {
    param([scriptblock]$Command)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command 2>$null } finally { $ErrorActionPreference = $previous }
}

function Test-CommandAvailable {
    param([string]$Command)
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

function Get-PortOwner {
    param([int]$Port)
    $line = netstat -ano | Select-String "LISTENING" | Select-String ":$Port " | Select-Object -First 1
    if (-not $line) { return $null }
    $ownerPid = $line.Line.Trim().Split()[-1]
    $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
    if ($proc) { return "$($proc.ProcessName) (PID $ownerPid)" }
    return "PID $ownerPid"
}

function Wait-ForUrl {
    param([string]$Url, [int]$TimeoutSeconds = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-WebRequest -UseBasicParsing $Url -TimeoutSec 3 -ErrorAction Stop
            return $true
        } catch { }
        Start-Sleep -Milliseconds 1500
    }
    return $false
}

# /ready answers 503 when degraded. Invoke-WebRequest in PowerShell 5.1 throws on that and
# loses the body, so use HttpClient, which returns the body for any status.
function Get-JsonBody {
    param([string]$Url, [int]$TimeoutSeconds = 60)
    Add-Type -AssemblyName System.Net.Http
    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
    try {
        return $client.GetAsync($Url).Result.Content.ReadAsStringAsync().Result
    } finally {
        $client.Dispose()
    }
}

Write-Host ""
Write-Host "=== AI Recruiter - Local Demo ===" -ForegroundColor Cyan
Write-Host ""

# ========== PREREQUISITES ==========
$missing = @()
if (-not (Test-CommandAvailable "python")) { $missing += "python" }
if (-not (Test-CommandAvailable "npm")) { $missing += "npm" }
if (-not $NoOllama -and -not (Test-CommandAvailable "ollama")) { $missing += "ollama (or pass -NoOllama)" }
if ($missing.Count -gt 0) {
    Write-Warn "Missing: $($missing -join ', ')"
    exit 1
}
foreach ($port in @($BackendPort, $FrontendPort)) {
    $owner = Get-PortOwner -Port $port
    if ($owner) {
        Write-Warn "Port $port is already in use by $owner. Stop it first or pass another port."
        exit 1
    }
}

# ========== OLLAMA ==========
if (-not $NoOllama) {
    Write-Step "Checking Ollama at $ollamaUrl ..."
    if (-not (Wait-ForUrl -Url "$ollamaUrl/api/tags" -TimeoutSeconds 2)) {
        Write-Info "Ollama is not running; starting 'ollama serve' ..."
        $null = Start-Process -WindowStyle Hidden -FilePath "ollama" -ArgumentList "serve" -PassThru
        if (-not (Wait-ForUrl -Url "$ollamaUrl/api/tags" -TimeoutSeconds 30)) {
            Write-Warn "Ollama did not come up within 30s. LLM features will use fallbacks."
        }
    }
    try {
        $tags = Invoke-RestMethod -Uri "$ollamaUrl/api/tags" -TimeoutSec 5
        $available = @($tags.models | ForEach-Object { $_.name })
        Write-Success "Ollama running ($($available.Count) models installed)"
        if (-not $SkipWarmup) {
            foreach ($model in $WarmupModels) {
                $installed = $available | Where-Object { $_ -eq $model -or $_ -like "${model}:*" }
                if (-not $installed) {
                    Write-Warn "Model $model is not installed. Run: ollama pull $model"
                    continue
                }
                # CPU-only first load takes ~25s per model; keep_alive holds it in RAM for the demo.
                Write-Info "Warming up $model (first load can take ~30s on CPU) ..."
                $body = @{ model = $model; prompt = "Say OK."; stream = $false; keep_alive = "60m"; options = @{ num_predict = 2 } } | ConvertTo-Json -Depth 3
                $watch = [Diagnostics.Stopwatch]::StartNew()
                try {
                    $null = Invoke-RestMethod -Uri "$ollamaUrl/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 240
                    Write-Success "$model warm ($([int]$watch.Elapsed.TotalSeconds)s)"
                } catch {
                    Write-Warn "Warm-up failed for ${model}: $($_.Exception.Message)"
                }
            }
        }
    } catch {
        Write-Warn "Ollama is unreachable: $($_.Exception.Message)"
    }
}

# ========== REDIS (optional) ==========
if (-not $NoRedis) {
    Write-Step "Checking Redis (Docker) ..."
    $dockerUp = $false
    if (Test-CommandAvailable "docker") {
        $null = Invoke-Native { docker info }
        $dockerUp = ($LASTEXITCODE -eq 0)
    }
    if (-not $dockerUp) {
        Write-Info "Docker Desktop is not running; the backend will use its in-memory CV queue."
    } else {
        $existing = Invoke-Native { docker ps -a --filter "name=^demo-redis$" --format "{{.Status}}" }
        if (-not $existing) {
            $null = Invoke-Native { docker run -d --name demo-redis -p 6379:6379 redis:7-alpine }
        } elseif ($existing -notlike "Up*") {
            $null = Invoke-Native { docker start demo-redis }
        }
        if ($LASTEXITCODE -eq 0) { Write-Success "Redis container demo-redis is running on 6379" }
        else { Write-Warn "Could not start demo-redis; the backend will use its in-memory CV queue." }
    }
}

# ========== BACKEND ==========
Write-Step "Starting backend on port $BackendPort ..."
# app/__init__.py already sets USE_TF=0; exporting it here also covers any other Python the backend spawns.
$env:USE_TF = "0"
$env:TRANSFORMERS_NO_TF = "1"
$beProcess = Start-Process -WindowStyle Hidden -PassThru -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort" `
    -WorkingDirectory $backendDir -RedirectStandardOutput $backendLog -RedirectStandardError "$backendLog.err"
# Startup loads the embedding model, so allow extra time.
if (-not (Wait-ForUrl -Url "$backendUrl/api/v1/health" -TimeoutSeconds 120)) {
    Write-Warn "Backend did not become healthy. Last log lines:"
    Get-Content "$backendLog.err" -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object { Write-Info $_ }
    if (-not $beProcess.HasExited) { Stop-Process -Id $beProcess.Id -Force }
    exit 1
}
Write-Success "Backend running at $backendUrl"

# ========== FRONTEND ==========
Write-Step "Starting frontend (npm run dev) on port $FrontendPort ..."
$env:VITE_API_PROXY_TARGET = $backendUrl
$feProcess = Start-Process -WindowStyle Hidden -PassThru -FilePath "npm.cmd" `
    -ArgumentList "run", "dev", "--", "--port", "$FrontendPort", "--strictPort" `
    -WorkingDirectory $frontendDir -RedirectStandardOutput $frontendLog -RedirectStandardError "$frontendLog.err"
# Vite binds to localhost, which may be IPv6 only ([::1]), so do not probe 127.0.0.1.
if (Wait-ForUrl -Url "http://localhost:$FrontendPort" -TimeoutSeconds 60) {
    Write-Success "Frontend running at http://localhost:$FrontendPort"
} else {
    Write-Warn "Frontend did not respond yet; check $frontendLog"
}

# ========== READINESS ==========
Write-Step "Readiness ($backendUrl/api/v1/ready):"
try {
    $ready = Get-JsonBody -Url "$backendUrl/api/v1/ready" | ConvertFrom-Json
    $color = if ($ready.status -eq "ok") { "Green" } else { "Red" }
    Write-Host "     overall: $($ready.status)" -ForegroundColor $color
    foreach ($check in $ready.checks.PSObject.Properties) {
        $c = if ($check.Value.status -eq "ok") { "Green" } else { "Red" }
        Write-Host ("     {0,-20} {1,-9} {2}" -f $check.Name, $check.Value.status, $check.Value.message) -ForegroundColor $c
    }
} catch {
    Write-Warn "Could not read /api/v1/ready: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "  App      : http://localhost:$FrontendPort" -ForegroundColor Cyan
Write-Host "  API docs : $backendUrl/docs" -ForegroundColor Cyan
Write-Host "  Logs     : $logDir (demo-*-$timestamp.log)" -ForegroundColor Cyan
Write-Host "  PIDs     : backend=$($beProcess.Id) frontend=$($feProcess.Id)" -ForegroundColor Cyan
Write-Host ""
if ($NoWait) {
    Write-Host "  Leaving services running. Stop them with:" -ForegroundColor Yellow
    Write-Host "    taskkill /PID $($beProcess.Id) /T /F; taskkill /PID $($feProcess.Id) /T /F"
    exit 0
}
Write-Host "  Press any key to stop the backend and frontend (Ollama and Redis keep running)." -ForegroundColor Yellow
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Step "Shutting down ..."
foreach ($proc in @($beProcess, $feProcess)) {
    if ($proc -and -not $proc.HasExited) {
        # npm spawns node children; /T stops the whole tree.
        $null = Invoke-Native { taskkill /PID $proc.Id /T /F }
    }
}
Write-Success "Stopped."
