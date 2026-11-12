@echo off
setlocal
cd /d "%~dp0"
title DocLens Setup

where pwsh.exe >nul 2>nul
if %errorlevel% equ 0 (
    pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
)

echo.
if %errorlevel% equ 0 (
    echo DocLens setup completed. You may now use START_DOCLENS.cmd.
) else (
    echo DocLens setup did not complete. Read the message above.
)
echo Press any key to close this window.
pause >nul
endlocal
