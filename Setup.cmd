@echo off
setlocal
cd /d "%~dp0"
title Setup

where pwsh.exe >nul 2>nul
if %errorlevel% equ 0 (
    pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\setup.ps1"
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\setup.ps1"
)

echo.
if %errorlevel% equ 0 (
    echo Setup completed. You may now double-click Start.cmd.
) else (
    echo Setup did not complete. Read the message above.
)
echo Press any key to close this window.
pause >nul
endlocal
