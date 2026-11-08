#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${WEIGHT_ANALYSIS_HOST_PYTHON:-/usr/bin/python3}" -S -m weight_analysis_scheduler.resources --snapshot "$@"
