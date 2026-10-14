#!/usr/bin/env bash
# One-command launcher for macOS / Linux: creates a venv on first run, then starts ANC.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "› Creating virtual environment (.venv)…"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
fi
if [ ! -f .venv/.deps-installed ] || [ requirements.txt -nt .venv/.deps-installed ]; then
  echo "› Installing dependencies…"
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.deps-installed
fi
if [ "$(uname)" = "Linux" ] && ! ldconfig -p 2>/dev/null | grep -q libportaudio; then
  echo "! PortAudio not found. Install it first, e.g.:  sudo apt install libportaudio2"
fi
exec .venv/bin/python -m anc "$@"
