@echo off
REM console code page stays default; this batch is ASCII-only.
REM Local Demucs service launcher - desktop acceleration mode.
REM The web page probes http://127.0.0.1:8001 and switches to local mode automatically.
REM
REM NOTE: torch does not support Python 3.14+ yet, so prefer 3.11/3.12/3.13
REM and create a project-local .venv virtual environment.
REM This file MUST stay ASCII-only with CRLF line endings: cmd.exe misparses
REM UTF-8 batch files containing Chinese after "chcp 65001" - commands get split.

setlocal
cd /d "%~dp0"

REM ---- 1) Find a Python version supported by torch - 3.11 / 3.12 / 3.13 ----
set "PYEXE="
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3.12"
if not defined PYEXE (
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 set "PYEXE=py -3.11"
)
if not defined PYEXE (
    py -3.13 --version >nul 2>&1
    if not errorlevel 1 set "PYEXE=py -3.13"
)
if not defined PYEXE (
    echo [ERROR] Python 3.11 / 3.12 / 3.13 not found.
    echo torch does not support Python 3.14 yet. Please install 3.12 - recommended.
    pause
    exit /b 1
)
echo Using Python: %PYEXE%

REM ---- 2) Prepare project-local virtual environment .venv ----
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv - first run only...
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

echo [1/2] Checking dependencies - first run installs torch, may take minutes...
"%VPY%" -c "import flask, flask_cors, demucs, torch, av" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies: flask flask-cors demucs ...
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Run manually:
        echo     "%VPY%" -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [2/2] Starting local Demucs service at http://127.0.0.1:8001 ...
echo Keep this window open; closing it stops the service.

REM Demucs / Whisper weights come from Hugging Face; use the CN mirror.
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"
REM hf-mirror does not support Xet; disable it or subtitle models return 401.
if not defined HF_HUB_DISABLE_XET set "HF_HUB_DISABLE_XET=1"
REM Keep HF cache inside the project to avoid ~/.cache permission issues.
if not defined HF_HOME set "HF_HOME=%~dp0.cache\hf"

REM torch.hub / demucs default cache can be unwritable in ~/.cache on some systems;
REM keep it inside the project as well (must match server.py TORCH_HOME layout).
if not defined TORCH_HOME set "TORCH_HOME=%~dp0.cache"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%" >nul 2>&1
if not exist "%HF_HOME%" mkdir "%HF_HOME%" >nul 2>&1

"%VPY%" server.py %*
pause
endlocal
