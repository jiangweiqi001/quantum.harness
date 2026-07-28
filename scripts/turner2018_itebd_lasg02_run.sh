#!/usr/bin/env bash

set -euo pipefail

: "${TURNER_ITEBD_STATE:?TURNER_ITEBD_STATE is required}"
case "$TURNER_ITEBD_STATE" in
  vacuum|Z2|Z3|Z4) ;;
  *)
    echo "unsupported TURNER_ITEBD_STATE=$TURNER_ITEBD_STATE" >&2
    exit 2
    ;;
esac

for variable in \
  SLURM_JOB_ID SLURM_JOB_PARTITION SLURM_JOB_ACCOUNT SLURM_JOB_QOS \
  SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE
do
  if [[ -z "${!variable:-}" ]]; then
    echo "$variable is required" >&2
    exit 2
  fi
done

require_scheduler_value() {
  local variable="$1"
  local expected="$2"
  local actual="${!variable}"
  if [[ "$actual" != "$expected" ]]; then
    echo "$variable=$actual does not match LASG02 Turner iTEBD (expected $expected)" >&2
    exit 2
  fi
}

require_scheduler_value SLURM_JOB_PARTITION ihicnormal
require_scheduler_value SLURM_JOB_ACCOUNT chenkun2025
require_scheduler_value SLURM_JOB_QOS user_student090
require_scheduler_value SLURM_JOB_NUM_NODES 1
require_scheduler_value SLURM_NTASKS 1
require_scheduler_value SLURM_CPUS_PER_TASK 8
if [[ "${SLURM_MEM_PER_NODE%M}" != 24000 ]]; then
  echo "SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE does not match LASG02 Turner iTEBD (expected 24000 MiB)" >&2
  exit 2
fi

readonly TURNER_SHARED_ROOT="/public/home/student090"
TURNER_REPO="${TURNER_REPO:-$TURNER_SHARED_ROOT/quantum.harness}"
TURNER_RUNTIME="${TURNER_RUNTIME:-$TURNER_SHARED_ROOT/python/cpython-3.12}"
readonly TURNER_DT005_OUTPUT_DIR="$TURNER_SHARED_ROOT/results/fig2-itebd"
readonly TURNER_DT0025_OUTPUT_DIR="$TURNER_SHARED_ROOT/results/fig2-itebd-dt0025"
TURNER_OUTPUT_DIR="${TURNER_OUTPUT_DIR:-$TURNER_DT005_OUTPUT_DIR}"
TURNER_PYTHON="${TURNER_PYTHON:-$TURNER_REPO/.venv-itebd/bin/python}"
TURNER_OFFICIAL_DATA="${TURNER_OFFICIAL_DATA:-$TURNER_REPO/.external/official-data/turner-2018}"
TURNER_ITEBD_TARGET_TIME="${TURNER_ITEBD_TARGET_TIME:-30.0}"
TURNER_ITEBD_DT="${TURNER_ITEBD_DT:-0.05}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$TURNER_REPO}"

if [[ "$TURNER_ITEBD_TARGET_TIME" != 30 && "$TURNER_ITEBD_TARGET_TIME" != 30.0 ]]; then
  echo "TURNER_ITEBD_TARGET_TIME must be 30.0" >&2
  exit 2
fi
case "$TURNER_ITEBD_DT" in
  0.05) ;;
  0.025)
    if [[ "$TURNER_OUTPUT_DIR" == "$TURNER_DT005_OUTPUT_DIR" ]]; then
      echo "dt=0.025 requires a distinct TURNER_OUTPUT_DIR (for example $TURNER_DT0025_OUTPUT_DIR)" >&2
      exit 2
    fi
    ;;
  *)
    echo "TURNER_ITEBD_DT must be 0.05 or 0.025" >&2
    exit 2
    ;;
esac

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
  TURNER_REPO TURNER_RUNTIME TURNER_OUTPUT_DIR TURNER_OFFICIAL_DATA \
  SLURM_SUBMIT_DIR
do
  require_approved_path "$variable"
done
canonical_python="$(realpath -m -- "$TURNER_PYTHON")"
case "$canonical_python" in
  "$approved_root"/*) ;;
  *)
    echo "TURNER_PYTHON must resolve below $TURNER_SHARED_ROOT: $TURNER_PYTHON" >&2
    exit 2
    ;;
esac
if [[ "$SLURM_SUBMIT_DIR" != "$TURNER_REPO" ]]; then
  echo "SLURM_SUBMIT_DIR must identify the reviewed TURNER_REPO checkout" >&2
  exit 2
fi

[[ -d "$TURNER_RUNTIME" ]] || {
  echo "missing offline CPython runtime: $TURNER_RUNTIME" >&2
  exit 2
}
[[ -x "$TURNER_PYTHON" ]] || {
  echo "missing offline virtual environment: $TURNER_PYTHON" >&2
  exit 2
}
[[ -d "$TURNER_OFFICIAL_DATA" ]] || {
  echo "missing official Turner data: $TURNER_OFFICIAL_DATA" >&2
  exit 2
}
checkpoint="$TURNER_OUTPUT_DIR/fig2_itebd_${TURNER_ITEBD_STATE}_checkpoint.h5"
[[ -f "$checkpoint" ]] || {
  echo "missing checkpoint for $TURNER_ITEBD_STATE: $checkpoint" >&2
  exit 2
}

expected_runtime="$(cd "$TURNER_RUNTIME" && pwd -P)"
actual_runtime="$(
  "$TURNER_PYTHON" -c \
    'import pathlib, sys; print(pathlib.Path(sys.base_prefix).resolve())'
)"
if [[ "$actual_runtime" != "$expected_runtime" ]]; then
  echo "offline virtual environment uses $actual_runtime, expected $expected_runtime" >&2
  exit 2
fi
"$TURNER_PYTHON" "$TURNER_REPO/scripts/turner2018_wheelhouse.py" \
  --manifest "$TURNER_REPO/scripts/turner2018_itebd_runtime_manifest.json" \
  --check-runtime

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

cd "$TURNER_REPO"
exec "$TURNER_PYTHON" -u scripts/turner2018_fig2_itebd.py \
  --state "$TURNER_ITEBD_STATE" \
  --target-time "$TURNER_ITEBD_TARGET_TIME" \
  --dt "$TURNER_ITEBD_DT" \
  --chi-max 400 \
  --sample-dt 0.1 \
  --checkpoint-dt 1.0 \
  --resume \
  --fit-start 0.0 \
  --fit-stop 12.0 \
  --output-dir "$TURNER_OUTPUT_DIR" \
  --official-data-dir "$TURNER_OFFICIAL_DATA"
