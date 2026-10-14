@echo off
REM One-command launcher for Windows: creates a venv on first run, then starts ANC.
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto deps
echo Creating virtual environment in .venv ...
py -3 -m venv .venv 2>nul
if not exist .venv\Scripts\python.exe python -m venv .venv
if not exist .venv\Scripts\python.exe (
  echo Python 3.10+ is required: https://www.python.org/downloads/
  exit /b 1
)
:deps
if exist .venv\.deps-installed goto run
echo Installing dependencies ...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
type nul > .venv\.deps-installed
:run
.venv\Scripts\python -m anc %*
