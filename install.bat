@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 Leon Priest (GitHub: 7h3v01d)
REM Chatterbox Studio installer: creates .venv, installs everything, picks the GPU build
REM of PyTorch when an NVIDIA card is present, then runs the install check.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  === Chatterbox Studio installer ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo  Python was not found. Install Python 3.11 from https://www.python.org/downloads/
    echo  and tick "Add python.exe to PATH" during setup, then run this again.
    pause & exit /b 1
)
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)"
if errorlevel 1 (
    for /f "delims=" %%v in ('python -c "import platform;print(platform.python_version())"') do set PYV=%%v
    echo  Python !PYV! found, but Chatterbox needs 3.10, 3.11 or 3.12 ^(3.11 recommended^).
    pause & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo  [1/4] Creating virtual environment...
    python -m venv .venv || (echo  Could not create .venv & pause & exit /b 1)
) else (
    echo  [1/4] Using existing .venv
)
set PY=.venv\Scripts\python.exe
"%PY%" -m pip install -q --upgrade pip setuptools wheel

echo  [2/4] Installing Chatterbox Studio and dependencies ^(several GB, be patient^)...
"%PY%" -m pip install -e ".[dev]" || (echo  Install failed - see the messages above. & pause & exit /b 1)

echo  [3/4] Checking for an NVIDIA GPU...
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo        No NVIDIA GPU found - using the CPU build ^(works, but slower^).
    goto check
)
for /f "delims=" %%v in ('"%PY%" -c "import torch;print(torch.__version__.split('+')[0])"') do set TV=%%v
for /f "delims=" %%v in ('"%PY%" -c "import torchaudio;print(torchaudio.__version__.split('+')[0])"') do set TAV=%%v
"%PY%" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" && (
    echo        CUDA build of PyTorch !TV! already installed.
    goto check
)
echo        NVIDIA GPU found - installing the CUDA build of PyTorch !TV!...
for %%c in (cu128 cu126 cu124 cu121) do (
    "%PY%" -m pip install -q --force-reinstall --no-deps torch==!TV! torchaudio==!TAV! --index-url https://download.pytorch.org/whl/%%c >nul 2>nul
    if not errorlevel 1 (
        "%PY%" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" && (
            echo        Installed torch !TV! for %%c.
            goto check
        )
    )
)
echo        Could not install a CUDA build automatically; continuing on CPU.
echo        See https://pytorch.org/get-started/locally/ to install it by hand.

:check
echo  [4/4] Checking the install...
"%PY%" -m chatterbox_studio doctor
echo.
echo  Done. Start the app with  run.bat
echo  ^(The first time you press Speak, the model downloads - a few GB, once.^)
pause
