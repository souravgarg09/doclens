[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$envFile = Join-Path $projectRoot ".env"
$envExample = Join-Path $projectRoot ".env.example"
$pythonExe = Join-Path $backendDir ".venv\Scripts\python.exe"

function Require-Command {
    param([string]$Name, [string]$InstallHint)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "$Name was not found. $InstallHint"
    }
    return $command
}

# The product name lives in brand.json; nothing else in this script hard-codes it.
$appName = "DocLens"
try { $appName = (Get-Content -LiteralPath (Join-Path $PSScriptRoot "brand.json") -Raw | ConvertFrom-Json).name } catch { }
Write-Host "$appName first-time setup" -ForegroundColor Cyan
Write-Host "This installs dependencies inside the extracted project folder." -ForegroundColor DarkGray

$uv = Get-Command "uv.exe" -ErrorAction SilentlyContinue
$npm = Require-Command "npm.cmd" "Install Node.js 20.19 or newer, then run this setup again."
$null = Require-Command "docker.exe" "Install and start Docker Desktop, then run this setup again."

$systemPythonExe = $null
$systemPythonArgs = @()
$pythonLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
if (-not $uv -and $pythonLauncher) {
    & $pythonLauncher.Source -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if ($LASTEXITCODE -eq 0) {
        $systemPythonExe = $pythonLauncher.Source
        $systemPythonArgs = @("-3.12")
    }
}
if (-not $uv -and -not $systemPythonExe) {
    $pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        & $pythonCommand.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) { $systemPythonExe = $pythonCommand.Source }
    }
}
if (-not $uv -and -not $systemPythonExe) {
    throw "Neither uv nor Python 3.12 was found. Install either one, reopen this window, and run setup again."
}

if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath $envExample -Destination $envFile
    Write-Host "Created .env from the safe template." -ForegroundColor Green
}

$environmentText = Get-Content -LiteralPath $envFile -Raw
if ($environmentText -match '(?m)^GEMINI_API_KEY=\s*$') {
    Write-Host "Add your own Gemini API key to GEMINI_API_KEY, save the file, then close Notepad." -ForegroundColor Yellow
    Start-Process -FilePath "notepad.exe" -ArgumentList @($envFile) -Wait
    $environmentText = Get-Content -LiteralPath $envFile -Raw
    if ($environmentText -match '(?m)^GEMINI_API_KEY=\s*$') {
        throw "GEMINI_API_KEY is still empty in .env. Add the key and run setup again."
    }
}

Write-Host "Creating the Python 3.12 environment..." -ForegroundColor Cyan
Push-Location $backendDir
try {
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        if ($uv) {
            # --relocatable keeps the environment working if the project folder is renamed or moved
            & $uv.Source venv .venv --python 3.12 --relocatable
        }
        else {
            & $systemPythonExe @systemPythonArgs -m venv .venv
        }
        if ($LASTEXITCODE -ne 0) { throw "Could not create the backend environment." }
    }
    if ($uv) {
        # --torch-backend=auto installs the CUDA build of PyTorch when an NVIDIA GPU is present
        & $uv.Source pip install --python $pythonExe --torch-backend=auto -r requirements.txt
    }
    else {
        & $pythonExe -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "pip could not be updated." }
        & $pythonExe -m pip install -r requirements.txt
    }
    if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }

    Write-Host "Downloading the AI models once (about 2 GB; later runs reuse them)..." -ForegroundColor Cyan
    & $pythonExe -m scripts.download_models
    if ($LASTEXITCODE -ne 0) { throw "Model download failed. Check the internet connection and run setup again." }
}
finally {
    Pop-Location
}

Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
Push-Location $frontendDir
try {
    & $npm.Source ci
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
}
finally {
    Pop-Location
}

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "You can now double-click Start.cmd."
