#!/usr/bin/env bash
# first_run: install the module's venv under $VIAM_MODULE_DATA, which survives module reloads.
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VIAM_MODULE_DATA:?must be set by viam-server}/venv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# uv, not python3 -m venv: Ubuntu often lacks python3-venv (nuc5 does). --allow-existing:
# viam-server re-runs first_run for each new module version.
uv venv --python 3.12 --allow-existing "$VENV"
VIRTUAL_ENV="$VENV" uv pip install -q -r requirements.txt
# Fail here, not at the first reconfigure: policy.py imports physicalai lazily.
"$VENV/bin/python" -c "import physicalai.inference, openvino; print('setup.sh: physicalai', openvino.__version__, 'OK')"
