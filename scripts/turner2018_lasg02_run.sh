#!/usr/bin/env bash

set -euo pipefail

: "${TURNER_ALLOWED_LENGTHS:?TURNER_ALLOWED_LENGTHS is required}"
: "${TURNER_LENGTH:?TURNER_LENGTH is required}"

for variable in \
  SLURM_JOB_PARTITION SLURM_JOB_ACCOUNT SLURM_JOB_QOS \
  SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK \
  SLURM_MEM_PER_NODE SLURM_TIMELIMIT
do
  if [[ -z "${!variable:-}" ]]; then
    echo "$variable is required" >&2
    exit 2
  fi
done

case "|$TURNER_ALLOWED_LENGTHS|" in
  *"|$TURNER_LENGTH|"*) ;;
  *)
    echo "TURNER_LENGTH=$TURNER_LENGTH is not allowed by this resource class ($TURNER_ALLOWED_LENGTHS)" >&2
    exit 2
    ;;
esac

case "$TURNER_LENGTH" in
  22|24|26|28|30) ;;
  *)
    echo "unsupported TURNER_LENGTH=$TURNER_LENGTH; LASG02 permits only 22,24,26,28,30" >&2
    exit 2
    ;;
esac

require_scheduler_value() {
  local variable="$1"
  local expected="$2"
  local actual="${!variable}"
  if [[ "$actual" != "$expected" ]]; then
    echo "$variable=$actual does not match LASG02 Turner ED (expected $expected)" >&2
    exit 2
  fi
}

require_scheduler_value SLURM_JOB_PARTITION ihicnormal
require_scheduler_value SLURM_JOB_ACCOUNT chenkun2025
require_scheduler_value SLURM_JOB_QOS user_student090
require_scheduler_value SLURM_JOB_NUM_NODES 1
require_scheduler_value SLURM_NTASKS 1
require_scheduler_value SLURM_CPUS_PER_TASK 24
if [[ "${SLURM_MEM_PER_NODE%M}" != 80000 ]]; then
  echo "SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE does not match LASG02 Turner ED (expected 80000 MiB)" >&2
  exit 2
fi
require_scheduler_value SLURM_TIMELIMIT 1440

readonly TURNER_SHARED_ROOT="/public/home/student090"
TURNER_REPO="${TURNER_REPO:-$TURNER_SHARED_ROOT/quantum.harness}"
TURNER_RUNTIME="${TURNER_RUNTIME:-$TURNER_SHARED_ROOT/python/cpython-3.12}"
TURNER_OUTPUT_DIR="${TURNER_OUTPUT_DIR:-$TURNER_SHARED_ROOT/results/turner-l${TURNER_LENGTH}}"
TURNER_PYTHON="${TURNER_PYTHON:-$TURNER_REPO/.venv/bin/python}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$TURNER_REPO}"

approved_root="$(realpath -m -- "$TURNER_SHARED_ROOT")"

require_approved_path() {
  local variable="$1"
  local value="${!variable}"
  local canonical
  canonical="$(realpath -m -- "$value")"
  case "$canonical" in
    "$approved_root"/*) printf -v "$variable" "%s" "$canonical" ;;
    *)
      echo "$variable must resolve below $TURNER_SHARED_ROOT: $value" >&2
      exit 2
      ;;
  esac
}

for variable in \
  TURNER_REPO TURNER_RUNTIME TURNER_OUTPUT_DIR TURNER_PYTHON SLURM_SUBMIT_DIR
do
  require_approved_path "$variable"
done
if [[ "$SLURM_SUBMIT_DIR" != "$TURNER_REPO" ]]; then
  echo "SLURM_SUBMIT_DIR must identify the reviewed TURNER_REPO checkout" >&2
  exit 2
fi

[[ -d "$TURNER_REPO" ]] || {
  echo "missing reviewed checkout: $TURNER_REPO" >&2
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
"$TURNER_PYTHON" "$TURNER_REPO/scripts/turner2018_wheelhouse.py" \
  --manifest "$TURNER_REPO/scripts/turner2018_wheelhouse_manifest.json" \
  --check-runtime

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
  --chunk-columns 24
