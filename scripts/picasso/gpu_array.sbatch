#!/usr/bin/env bash
#SBATCH --job-name=wa-geometry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
set -Eeuo pipefail
trap 'echo "[ERROR] GPU array line=$LINENO exit=$? task=${SLURM_ARRAY_TASK_ID:-?}" >&2' ERR
module load pytorch/2.10.0
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH" APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
RUN_DIR="${1:?run directory required}"
ARRAY_MAP="${2:?array map required}"
: "${SLURM_ARRAY_TASK_ID:?submit using the Python submit command}"
echo "[GPU] job=${SLURM_JOB_ID:-none} task=$SLURM_ARRAY_TASK_ID host=$(hostname) run=$RUN_DIR"
# The shell receives SLURM's early signal and forwards it to the Python worker.
worker_pid=''
trap 'if [[ -n "$worker_pid" ]]; then kill -USR1 "$worker_pid" 2>/dev/null || true; fi' USR1
python -m weight_analysis.experiments worker --run-dir "$RUN_DIR" --array-map "$ARRAY_MAP" \
  --array-index "$SLURM_ARRAY_TASK_ID" --device gpu --budget-seconds 27000 &
worker_pid=$!
# wait can return >128 after the shell trap; keep waiting for the worker itself.
set +e
while true; do
  wait "$worker_pid"
  worker_status=$?
  if kill -0 "$worker_pid" 2>/dev/null; then continue; fi
  break
done
set -e
exit "$worker_status"
