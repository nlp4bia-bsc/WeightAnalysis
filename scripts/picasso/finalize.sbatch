#!/usr/bin/env bash
#SBATCH --job-name=wa-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu
#SBATCH --time=08:00:00
set -Eeuo pipefail
trap 'echo "[ERROR] finalize line=$LINENO exit=$?" >&2' ERR
module load pytorch/2.10.0
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH" APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "[FINALIZE] job=${SLURM_JOB_ID:-none} host=$(hostname) run=${1:?run directory required}"
python -m weight_analysis.experiments finalize --run-dir "$1"
