#!/usr/bin/env bash
set -euo pipefail

SERVING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SERVING_PYTHON:-$SERVING_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONPATH="$SERVING_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m vla_serving.cli "$@"
