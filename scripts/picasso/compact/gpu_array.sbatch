#!/usr/bin/env bash
#SBATCH --job-name=wa3-geometry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
source "${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}/scripts/picasso/compact/common.sh"
RUN_DIR="${1:?Run directory required}"
ARRAY_MAP="${2:?Array map required}"
: "${SLURM_ARRAY_TASK_ID:?Array submission required}"
worker_pid=''
trap 'if [[ -n "$worker_pid" ]]; then kill -USR1 "$worker_pid" 2>/dev/null || true; fi' USR1
python -m weight_analysis.experiments.compact worker --run-dir "$RUN_DIR" --array-map "$ARRAY_MAP" \
  --array-index "$SLURM_ARRAY_TASK_ID" --device gpu --budget-seconds 27000 &
worker_pid=$!
set +e
while true; do
  wait "$worker_pid"
  worker_status=$?
  if kill -0 "$worker_pid" 2>/dev/null; then continue; fi
  break
done
set -e
exit "$worker_status"
