@echo off
REM Start the Chatterbox Studio desktop app.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Not installed yet - running install.bat first.
    call install.bat
)
start "" ".venv\Scripts\pythonw.exe" -m chatterbox_studio gui %*
