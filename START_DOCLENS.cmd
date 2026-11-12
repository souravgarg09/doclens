@echo off
setlocal
cd /d "%~dp0"
title DocLens Launcher

where pwsh.exe >nul 2>nul
if %errorlevel% equ 0 (
    pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
)

if %errorlevel% neq 0 (
    echo.
    echo DocLens could not start. Read the message above, then press any key.
    pause >nul
)
endlocal
