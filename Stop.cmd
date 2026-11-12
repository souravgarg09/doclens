@echo off
setlocal
cd /d "%~dp0"
title Stop

where pwsh.exe >nul 2>nul
if %errorlevel% equ 0 (
    pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\stop.ps1"
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\stop.ps1"
)

echo.
pause
endlocal
