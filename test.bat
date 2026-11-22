@echo off
REM Run the test suite (uses a fake model: no GPU or download needed).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pytest -q %*
