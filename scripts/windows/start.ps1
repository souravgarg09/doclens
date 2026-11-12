[CmdletBinding()]
param(
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# The product name lives in brand.json; nothing else in this script hard-codes it.
$appName = "DocLens"
try { $appName = (Get-Content -LiteralPath (Join-Path $PSScriptRoot "..\..\brand.json") -Raw | ConvertFrom-Json).name } catch { }

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path  # scripts\windows -> project root
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$pythonExe = Join-Path $backendDir ".venv\Scripts\python.exe"
$runtimeDir = Join-Path $projectRoot ".runtime"
$logDir = Join-Path $runtimeDir "logs"
$pidFile = Join-Path $runtimeDir "pids.json"

function Test-TcpPort {
    param([int]$Port)
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        return $client.ConnectAsync("127.0.0.1", $Port).Wait(300)
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Wait-HttpEndpoint {
    param([string]$Uri, [int]$TimeoutSeconds = 45)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Backend environment is missing. Complete the first-time setup in README.md."
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".env"))) {
    throw ".env is missing. Copy .env.example to .env and add GEMINI_API_KEY."
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendDir "node_modules"))) {
    throw "Frontend dependencies are missing. Run 'npm ci' once inside frontend."
}

$docker = Get-Command docker -ErrorAction SilentlyContinue
$dockerCandidates = @(
    (Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\docker.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe")
)
$dockerExe = if ($docker) { $docker.Source } else { $dockerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1 }
$dockerDesktopCandidates = @(
    (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe")
)
$dockerDesktopExe = $dockerDesktopCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) { throw "npm was not found. Install Node.js or add it to PATH." }

$occupied = @(@(5173, 8000) | Where-Object { Test-TcpPort $_ })
if ($occupied.Count -gt 0) {
    throw "$appName is already running, or required port(s) are occupied: $($occupied -join ', '). Run Stop.cmd first if these are stale processes from an earlier run."
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not ((Test-TcpPort 5432) -and (Test-TcpPort 6379) -and (Test-TcpPort 6333))) {
    if ($dockerDesktopExe -and -not (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue)) {
        Write-Host "Starting Docker Desktop..." -ForegroundColor Cyan
        Start-Process -FilePath $dockerDesktopExe -WindowStyle Hidden
    }
}

Write-Host "Starting PostgreSQL, Redis, and Qdrant..." -ForegroundColor Cyan
if ($dockerExe) {
    $engineDeadline = (Get-Date).AddSeconds(90)
    do {
        & $dockerExe info *> $null
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $engineDeadline)
    if ($LASTEXITCODE -ne 0) { throw "Docker Desktop did not become ready within 90 seconds." }
    & $dockerExe compose --project-directory $projectRoot up -d
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose could not start the infrastructure." }
}
else {
    Write-Host "Docker CLI is not visible in this terminal; waiting for Docker Desktop's existing containers..." -ForegroundColor Yellow
}

$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline -and -not ((Test-TcpPort 5432) -and (Test-TcpPort 6379) -and (Test-TcpPort 6333))) {
    Start-Sleep -Milliseconds 500
}
if (-not ((Test-TcpPort 5432) -and (Test-TcpPort 6379) -and (Test-TcpPort 6333))) {
    throw "Docker services did not become ready. Open Docker Desktop, start the containers, and try again."
}

Write-Host "Starting API, parsing worker, and frontend (the first start after boot can take a minute)..." -ForegroundColor Cyan
$api = Start-Process -FilePath $pythonExe -WorkingDirectory $backendDir -WindowStyle Hidden -PassThru `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -RedirectStandardOutput (Join-Path $logDir "api.out.log") `
    -RedirectStandardError (Join-Path $logDir "api.err.log")
$worker = Start-Process -FilePath $pythonExe -WorkingDirectory $backendDir -WindowStyle Hidden -PassThru `
    -ArgumentList @("-m", "workers.parse_worker") `
    -RedirectStandardOutput (Join-Path $logDir "worker.out.log") `
    -RedirectStandardError (Join-Path $logDir "worker.err.log")
$frontend = Start-Process -FilePath $npm.Source -WorkingDirectory $frontendDir -WindowStyle Hidden -PassThru `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") `
    -RedirectStandardOutput (Join-Path $logDir "frontend.out.log") `
    -RedirectStandardError (Join-Path $logDir "frontend.err.log")

@{
    api = $api.Id
    worker = $worker.Id
    frontend = $frontend.Id
    started_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8

try {
    if (-not (Wait-HttpEndpoint "http://127.0.0.1:8000/api/health" 180)) {
        throw "The API did not become ready. See .runtime\logs\api.err.log."
    }
    if (-not (Wait-HttpEndpoint "http://127.0.0.1:5173/" 30)) {
        throw "The frontend did not become ready. See .runtime\logs\frontend.err.log."
    }
}
catch {
    & (Join-Path $PSScriptRoot "stop.ps1") -KeepInfrastructure
    throw
}

Write-Host "$appName is ready: http://127.0.0.1:5173/" -ForegroundColor Green
Write-Host "The worker loads its AI models in the background; the page shows when it is ready."
Write-Host "Logs: $logDir"
Write-Host "Stop everything later with: Stop.cmd"
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:5173/"
}
