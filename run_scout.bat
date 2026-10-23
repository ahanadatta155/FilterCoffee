@echo off
cd /d "%~dp0"
uv run python run_scout.py %*
if errorlevel 1 echo Nemotron Scout exited with an error.
pause
