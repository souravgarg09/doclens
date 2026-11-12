[CmdletBinding()]
param(
    [switch]$KeepInfrastructure
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$pidFile = Join-Path $projectRoot ".runtime\pids.json"

if (Test-Path -LiteralPath $pidFile) {
    $saved = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    foreach ($name in @("frontend", "worker", "api")) {
        $processId = [int]$saved.$name
        if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
            & taskkill.exe /PID $processId /T /F 2>$null | Out-Null
            Write-Host "Stopped $name." -ForegroundColor DarkGray
        }
    }
    Remove-Item -LiteralPath $pidFile -Force
}
else {
    Write-Host "No DocLens PID file was found; application processes may already be stopped."
}

if (-not $KeepInfrastructure) {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    $dockerCandidates = @(
        (Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\docker.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe")
    )
    $dockerExe = if ($docker) { $docker.Source } else { $dockerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1 }
    if ($dockerExe) {
        & $dockerExe compose --project-directory $projectRoot stop
        if ($LASTEXITCODE -ne 0) { throw "Docker Compose could not stop the infrastructure." }
        Write-Host "Stopped Docker services. Stored data was preserved." -ForegroundColor DarkGray
    }
    else {
        Write-Host "Docker CLI was not visible; application processes are stopped, but Docker services were left running." -ForegroundColor Yellow
    }
}

Write-Host "DocLens is stopped." -ForegroundColor Green
