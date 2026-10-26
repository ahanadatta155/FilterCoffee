@echo off
cd /d "%~dp0"
uv run python -m scout.web
if errorlevel 1 (
    echo Scout could not start. Check that uv is installed and port 8765 is available.
    pause
)
