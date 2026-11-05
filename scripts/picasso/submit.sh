#!/usr/bin/env bash
# Invoke from a login node after the CPU preparation job has frozen plan.json.
set -Eeuo pipefail
module load pytorch/2.10.0
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
python -m weight_analysis.experiments submit --scripts "$PROJECT_DIR/scripts/picasso" "$@"
