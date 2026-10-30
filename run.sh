#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${VIAM_MODULE_DATA:?must be set by viam-server}/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "run.sh: no venv at $PYTHON -- first_run (setup.sh) did not complete" >&2
  exit 1
fi
exec "$PYTHON" -m src.main "$@"
