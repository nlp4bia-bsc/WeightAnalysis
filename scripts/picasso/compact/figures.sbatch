#!/usr/bin/env bash
#SBATCH --job-name=wa3-figures
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu
#SBATCH --time=08:00:00
source "${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}/scripts/picasso/compact/common.sh"
python -m weight_analysis.experiments.compact figures --run-dir "${1:?Run directory required}"
