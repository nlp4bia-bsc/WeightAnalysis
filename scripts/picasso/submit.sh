#!/usr/bin/env bash
# Invoke from a login node after the CPU preparation job has frozen plan.json.
set -Eeuo pipefail
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# The module's python is a container launcher: sbatch only exists on the host.
# No scientific dependencies or site packages are needed by this adapter.
HOST_PYTHON="${WEIGHT_ANALYSIS_HOST_PYTHON:-/usr/bin/python3}"
if [[ ! -x "$HOST_PYTHON" ]]; then
  echo "[ERROR] Host Python not found: $HOST_PYTHON; set WEIGHT_ANALYSIS_HOST_PYTHON to its absolute path" >&2
  exit 127
fi
echo "[SUBMIT HOST] python=$HOST_PYTHON project=$PROJECT_DIR"
exec "$HOST_PYTHON" -S -m weight_analysis_scheduler --scripts "$PROJECT_DIR/scripts/picasso" "$@"
