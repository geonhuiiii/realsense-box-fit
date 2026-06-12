#!/usr/bin/env bash
# Start the web app with the project venv (Python 3.10+ required for SAM2 + Gemma 4).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${RBF_PYTHON:-$ROOT/.venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "[RBF] no venv at $ROOT/.venv — creating with Homebrew python3.10 …"
  CAND="${HOMEBREW_PREFIX:-/opt/homebrew}/bin/python3.10"
  [ -x "$CAND" ] || CAND="$(command -v python3.10 || true)"
  [ -n "$CAND" ] || { echo "ERROR: install Python 3.10+ (brew install python@3.10)"; exit 1; }
  "$CAND" -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install --upgrade pip
  "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements-full.txt"
  "$ROOT/.venv/bin/pip" install 'git+https://github.com/facebookresearch/segment-anything-2.git'
  PY="$ROOT/.venv/bin/python"
fi

exec "$PY" "$ROOT/app.py" "$@"
