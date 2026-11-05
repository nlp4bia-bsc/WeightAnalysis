#!/usr/bin/env bash
#SBATCH --job-name=wa-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=wa-tests-%j.out
#SBATCH --error=wa-tests-%j.err
set -Eeuo pipefail
trap 'echo "[ERROR] tests line=$LINENO exit=$?" >&2' ERR
module load pytorch/2.10.0
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH" APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "[TESTS] job=${SLURM_JOB_ID:-none} host=$(hostname) project=$PROJECT_DIR"
python - <<'PY'
import sys
import torch
import pyarrow
print("Python:", sys.version, "PyTorch:", torch.__version__, "PyArrow:", pyarrow.__version__, flush=True)
assert torch.cuda.is_available(), "CUDA unavailable in the DGX test allocation"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY
BUNDLE_DIR="${1:?Use bash scripts/picasso/submit_tests.sh to build the installed test package first}"
shift
python -m weight_analysis.maintenance.testing run --project-dir "$PROJECT_DIR" \
  --bundle-dir "$BUNDLE_DIR" --require-cuda -- -q -rs "$@"
