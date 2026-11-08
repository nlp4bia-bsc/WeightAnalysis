#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}"
exec bash "$PROJECT_DIR/scripts/picasso/compact/submit.sh" --retry "$@"
