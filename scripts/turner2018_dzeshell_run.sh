#!/usr/bin/env bash

set -euo pipefail

: "${TURNER_ALLOWED_LENGTHS:?TURNER_ALLOWED_LENGTHS is required}"
: "${TURNER_LENGTH:?TURNER_LENGTH is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
: "${SLURM_MEM_PER_NODE:?SLURM_MEM_PER_NODE is required}"
: "${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is required}"

case "|$TURNER_ALLOWED_LENGTHS|" in
  *"|$TURNER_LENGTH|"*) ;;
  *)
    echo "TURNER_LENGTH=$TURNER_LENGTH is not allowed by this resource class ($TURNER_ALLOWED_LENGTHS)" >&2
    exit 2
    ;;
esac

case "$TURNER_LENGTH" in
  22|24|26|28)
    expected_cpus=8
    expected_gpus=1
    expected_memory=60000
    ;;
  30)
    expected_cpus=16
    expected_gpus=2
    expected_memory=120000
    ;;
  32)
    expected_cpus=32
    expected_gpus=4
    expected_memory=240000
    ;;
  *)
    echo "unsupported TURNER_LENGTH=$TURNER_LENGTH" >&2
    exit 2
    ;;
esac

if [[ "$SLURM_CPUS_PER_TASK" != "$expected_cpus" ]]; then
  echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK does not match L=$TURNER_LENGTH (expected $expected_cpus)" >&2
  exit 2
fi
if [[ "${SLURM_MEM_PER_NODE%M}" != "$expected_memory" ]]; then
  echo "SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE does not match L=$TURNER_LENGTH (expected ${expected_memory}M)" >&2
  exit 2
fi
if [[ "$SLURM_GPUS_ON_NODE" != "$expected_gpus" ]]; then
  echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE does not match L=$TURNER_LENGTH (expected $expected_gpus)" >&2
  exit 2
fi

TURNER_SHARED_ROOT="${TURNER_SHARED_ROOT:-/work/share/giggleliu/jiangweiqi}"
TURNER_REPO="${TURNER_REPO:-$TURNER_SHARED_ROOT/quantum.harness}"
TURNER_RUNTIME="${TURNER_RUNTIME:-$TURNER_SHARED_ROOT/python/cpython-3.12}"
TURNER_OUTPUT_DIR="${TURNER_OUTPUT_DIR:-$TURNER_SHARED_ROOT/results/turner-l${TURNER_LENGTH}}"
TURNER_PYTHON="${TURNER_PYTHON:-$TURNER_REPO/.venv/bin/python}"

[[ -d "$TURNER_REPO" ]] || {
  echo "missing shared checkout: $TURNER_REPO" >&2
  exit 2
}
[[ -d "$TURNER_RUNTIME" ]] || {
  echo "missing offline CPython runtime: $TURNER_RUNTIME" >&2
  exit 2
}
[[ -x "$TURNER_PYTHON" ]] || {
  echo "missing offline virtual environment: $TURNER_PYTHON" >&2
  exit 2
}

expected_runtime="$(cd "$TURNER_RUNTIME" && pwd -P)"
actual_runtime="$(
  "$TURNER_PYTHON" -c \
    'import pathlib, sys; print(pathlib.Path(sys.base_prefix).resolve())'
)"
[[ "$actual_runtime" == "$expected_runtime" ]] || {
  echo "offline virtual environment uses $actual_runtime, expected $expected_runtime" >&2
  exit 2
}
"$TURNER_PYTHON" -c \
  'import sys; assert sys.version_info[:2] == (3, 12); import h5py, numpy, scipy'

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p "$TURNER_OUTPUT_DIR"
cd "$TURNER_REPO"
exec "$TURNER_PYTHON" -u scripts/turner2018_l32_server.py \
  --length "$TURNER_LENGTH" \
  --stage all \
  --output-dir "$TURNER_OUTPUT_DIR" \
  --chunk-columns 32
