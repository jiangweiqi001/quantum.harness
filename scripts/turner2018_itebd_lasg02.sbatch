#!/usr/bin/env bash
#SBATCH --job-name=turner-itebd
#SBATCH --partition=ihicnormal
#SBATCH --account=chenkun2025
#SBATCH --qos=user_student090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=24:00:00
#SBATCH --output=turner-itebd-%x-%j.out

set -euo pipefail

: "${TURNER_ITEBD_STATE:?set TURNER_ITEBD_STATE to vacuum, Z2, Z3, or Z4}"
case "$TURNER_ITEBD_STATE" in
  vacuum|Z2|Z3|Z4) ;;
  *)
    echo "unsupported TURNER_ITEBD_STATE=$TURNER_ITEBD_STATE" >&2
    exit 2
    ;;
esac
export TURNER_ITEBD_STATE

TURNER_SHARED_ROOT="/public/home/student090"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
canonical_root="$(realpath -e -- "$TURNER_SHARED_ROOT")"
canonical_submit_dir="$(realpath -e -- "$SLURM_SUBMIT_DIR")"
expected_submit_dir="$canonical_root/quantum.harness"
if [[ "$canonical_submit_dir" != "$expected_submit_dir" ]]; then
  echo "SLURM_SUBMIT_DIR must resolve exactly to $expected_submit_dir" >&2
  exit 2
fi
expected_runner="$canonical_submit_dir/scripts/turner2018_itebd_lasg02_run.sh"
canonical_runner="$(realpath -e -- "$expected_runner")"
if [[ "$canonical_runner" != "$expected_runner" || ! -f "$canonical_runner" ]]; then
  echo "LASG02 iTEBD runner must be the canonical checkout file" >&2
  exit 2
fi

source "$canonical_runner"
