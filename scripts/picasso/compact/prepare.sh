#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="${WEIGHT_ANALYSIS_PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export WEIGHT_ANALYSIS_PROJECT_DIR="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR="${1:?Usage: prepare.sh NEW_RUN_DIR [prepare options]}"
shift
HOST_PYTHON="${WEIGHT_ANALYSIS_HOST_PYTHON:-/usr/bin/python3}"
RUN_DIR="$(realpath -m -- "$RUN_DIR")"
if [[ -e "$RUN_DIR/plan.json" || -d "$RUN_DIR/tasks" || -e "$RUN_DIR/manifest/plan.json" ]]; then
  echo "[ERROR] Existing run detected; use a NEW run directory or resume its existing tasks" >&2
  exit 2
fi
original=("$@")
resources=()
for ((i=0; i<${#original[@]}; i++)); do
  case "${original[$i]}" in
    --allow-unknown-quota) resources+=("${original[$i]}");;
    --quota-file|--reserve-gib|--reserve-inodes|--max-run-files)
      resources+=("${original[$i]}" "${original[$((i+1))]:?Missing resource option value}"); i=$((i+1));;
  esac
done
"$HOST_PYTHON" -S -m weight_analysis_scheduler.resources --root "$RUN_DIR" "${resources[@]}"
mkdir -p "$RUN_DIR/logs"
sbatch --chdir="$PWD" --output="$RUN_DIR/logs/prepare-%j.out" --error="$RUN_DIR/logs/prepare-%j.err" \
  "$PWD/scripts/picasso/compact/prepare.sbatch" --run-dir "$RUN_DIR" "$@"
