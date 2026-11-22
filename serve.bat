@echo off
REM Run only the HTTP API (no window). Extra args are passed through, e.g.
REM   serve.bat --preload
REM   serve.bat --host 0.0.0.0 --token YOUR-LONG-RANDOM-TOKEN
cd /d "%~dp0"
".venv\Scripts\python.exe" -m chatterbox_studio serve %*
