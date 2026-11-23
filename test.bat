@echo off
REM Run the test suite (fake model: no GPU or download needed).
REM CBX_STRICT=1 makes any skipped test fail the run, so missing dependencies can't hide.
cd /d "%~dp0"
set CBX_STRICT=1
".venv\Scripts\python.exe" -m pytest -q %*
