#!/usr/bin/env bash
# Build on the login node, then install/test that wheel in the allocation's Python.
set -Eeuo pipefail
module load pytorch/2.10.0
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH" APPTAINERENV_PYTHONPATH="$PYTHONPATH"
mkdir -p "$PROJECT_DIR/local_results/test_packages"
BUNDLE_DIR=$(mktemp -d "$PROJECT_DIR/local_results/test_packages/test-XXXXXX")
python -m weight_analysis.maintenance.testing build --project-dir "$PROJECT_DIR" --output-dir "$BUNDLE_DIR"
sbatch "$PROJECT_DIR/scripts/picasso/tests.sbatch" "$BUNDLE_DIR" "$@"
