#!/usr/bin/env bash
#SBATCH --job-name=wa-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=wa-prepare-%j.out
#SBATCH --error=wa-prepare-%j.err
set -Eeuo pipefail
trap 'echo "[ERROR] prepare line=$LINENO exit=$?" >&2' ERR
module load pytorch/2.10.0
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH" APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "[PREPARE] job=${SLURM_JOB_ID:-none} host=$(hostname) project=$PROJECT_DIR"
python -m weight_analysis.experiments prepare "$@"
