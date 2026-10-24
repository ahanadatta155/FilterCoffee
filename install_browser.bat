@echo off
cd /d "%~dp0"
uv sync
if errorlevel 1 (
    echo Failed to install project dependencies.
    pause
    exit /b 1
)
uv run playwright install chromium
if errorlevel 1 (
    echo Failed to install Playwright Chromium.
    pause
    exit /b 1
)
echo Chromium is ready for Nemotron Scout.
pause
