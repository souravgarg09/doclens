[CmdletBinding()]
param(
    [switch]$KeepInfrastructure
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# The product name lives in brand.json; nothing else in this script hard-codes it.
$appName = "DocLens"
try { $appName = (Get-Content -LiteralPath (Join-Path $PSScriptRoot "..\..\brand.json") -Raw | ConvertFrom-Json).name } catch { }

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path  # scripts\windows -> project root
$pidFile = Join-Path $projectRoot ".runtime\pids.json"
# What each of our processes looks like on its command line.
$markers = @{ api = "uvicorn"; worker = "workers.parse_worker"; frontend = "npm" }

function Get-ProcessInfo([int]$ProcessId) {
    Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Test-OurProcess($info, [string]$marker, $startedAt) {
    # A PID is only trusted if the process still looks like the one we started:
    # Windows reuses PIDs, so after a reboot the same number can belong to anything.
    if (-not $info -or -not $info.CommandLine) { return $false }
    if ($info.CommandLine -notlike "*$marker*") { return $false }
    if ($startedAt) {
        # started within two minutes of the launcher run that recorded it
        $delta = [math]::Abs(($info.CreationDate - $startedAt).TotalSeconds)
        if ($delta -gt 120) { return $false }
    }
    return $true
}

function Test-InProject($info) {
    # True when the process, or its parent, runs from this project folder. The parent check
    # matters on Windows: backend\.venv\Scripts\python.exe is a small launcher that starts
    # the real interpreter (outside the project) as its child.
    if (-not $info) { return $false }
    foreach ($candidate in @($info, (Get-ProcessInfo $info.ParentProcessId))) {
        if (-not $candidate) { continue }
        if ($candidate.ExecutablePath -and $candidate.ExecutablePath -like "$projectRoot\*") { return $true }
        if ($candidate.CommandLine -and $candidate.CommandLine -like "*$projectRoot\*") { return $true }
    }
    return $false
}

function Stop-Tree([int]$ProcessId, [string]$name) {
    & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null
    Write-Host "Stopped $name." -ForegroundColor DarkGray
}

$stoppedAny = $false
if (Test-Path -LiteralPath $pidFile) {
    $saved = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    # PowerShell 7 already turns the ISO timestamp into a DateTime; Windows PowerShell 5
    # leaves a string. Never re-parse with the local culture (dd-MM vs MM-dd mix-ups).
    $startedAt = $null
    if ($saved.started_at -is [datetime]) { $startedAt = $saved.started_at }
    elseif ($saved.started_at) {
        try { $startedAt = [datetime]::Parse($saved.started_at, [Globalization.CultureInfo]::InvariantCulture,
                                             [Globalization.DateTimeStyles]::RoundtripKind) } catch { }
    }
    foreach ($name in @("frontend", "worker", "api")) {
        $processId = [int]$saved.$name
        $info = Get-ProcessInfo $processId
        if (Test-OurProcess $info $markers[$name] $startedAt) {
            Stop-Tree $processId $name
            $stoppedAny = $true
        }
        elseif ($info) {
            Write-Host "PID $processId now belongs to another program; left it alone." -ForegroundColor Yellow
        }
    }
    Remove-Item -LiteralPath $pidFile -Force
}

# Orphans (e.g. the PID file was deleted): stop whatever of ours still holds the ports.
foreach ($entry in @(@{ port = 8000; name = "api" }, @{ port = 5173; name = "frontend" })) {
    $owners = Get-NetTCPConnection -LocalPort $entry.port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
        $info = Get-ProcessInfo $owner
        # Only processes started from THIS project folder: another copy of the project
        # (or any other app) on the same port must never be touched.
        if (Test-InProject $info) {
            Stop-Tree $owner "$($entry.name) (orphaned)"
            $stoppedAny = $true
        }
    }
}
# The worker holds no port; find it by its command line inside this project.
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*workers.parse_worker*" -and (Test-InProject $_) } |
    ForEach-Object { Stop-Tree $_.ProcessId "worker (orphaned)"; $stoppedAny = $true }

if (-not $stoppedAny) {
    Write-Host "No running $appName processes were found."
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

Write-Host "$appName is stopped." -ForegroundColor Green
