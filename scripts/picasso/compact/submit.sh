#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export WEIGHT_ANALYSIS_PROJECT_DIR="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
HOST_PYTHON="${WEIGHT_ANALYSIS_HOST_PYTHON:-/usr/bin/python3}"
exec "$HOST_PYTHON" -S -m weight_analysis_scheduler --scripts "$PWD/scripts/picasso/compact" "$@"
