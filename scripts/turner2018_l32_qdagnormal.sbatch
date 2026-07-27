#!/usr/bin/env bash
#SBATCH --job-name=turner-l32-ed-fsa
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A800:1
#SBATCH --output=turner-l32-%j.out

set -euo pipefail

export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export BLIS_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# The qdeshell profile is offline. Point this at a pre-staged Apptainer image;
# do not install packages or place credentials in this script.
TURNER_OFFLINE_IMAGE="${TURNER_OFFLINE_IMAGE:-}"
TURNER_OUTPUT_DIR="${TURNER_OUTPUT_DIR:-$HOME/quantum.harness/results/turner-l32}"
TURNER_REPO="${TURNER_REPO:-$HOME/quantum.harness}"
TURNER_PYTHON="${TURNER_PYTHON:-$TURNER_REPO/.venv/bin/python}"

mkdir -p "$TURNER_OUTPUT_DIR"
cd "$TURNER_REPO"

args=(
  scripts/turner2018_l32_server.py
  --length 32
  --stage all
  --output-dir "$TURNER_OUTPUT_DIR"
  --chunk-columns 32
)

if [[ -n "$TURNER_OFFLINE_IMAGE" ]]; then
  exec apptainer exec --cleanenv \
    --env OMP_NUM_THREADS=64,OPENBLAS_NUM_THREADS=64,MKL_NUM_THREADS=64 \
    --bind "$TURNER_REPO:$TURNER_REPO" \
    --bind "$TURNER_OUTPUT_DIR:$TURNER_OUTPUT_DIR" \
    "$TURNER_OFFLINE_IMAGE" python -u "${args[@]}"
fi

[[ -x "$TURNER_PYTHON" ]] || {
  echo "missing offline Python environment: $TURNER_PYTHON" >&2
  echo "set TURNER_OFFLINE_IMAGE or pre-stage TURNER_PYTHON" >&2
  exit 2
}
exec "$TURNER_PYTHON" -u "${args[@]}"
