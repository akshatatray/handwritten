#!/bin/bash
# Creates (or refreshes) the skill's private Python environment at ~/.handwritten/.venv.
# Safe to run every time: it does nothing when requirements haven't changed.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_HOME="${HANDWRITTEN_HOME:-$HOME/.handwritten}"
VENV="$HW_HOME/.venv"
REQ="$SKILL_DIR/requirements.txt"
STAMP="$VENV/.requirements.sha"

if [ ! -x "$VENV/bin/python" ]; then
  PY=""
  for cand in python3.13 python3.12 python3.11 python3.10 /usr/bin/python3 python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; assert sys.version_info >= (3, 9)' >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "ERROR: no working Python 3.9+ found" >&2; exit 1
  fi
  mkdir -p "$HW_HOME"
  "$PY" -m venv "$VENV"
fi

WANT="$(shasum "$REQ" | cut -d' ' -f1)"
if [ "$(cat "$STAMP" 2>/dev/null || true)" != "$WANT" ]; then
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -r "$REQ"
  echo "$WANT" > "$STAMP"
fi
echo "ready: $VENV/bin/python"
