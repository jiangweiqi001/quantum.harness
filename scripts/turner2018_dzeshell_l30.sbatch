#!/usr/bin/env bash
#SBATCH --job-name=turner-ed-l30
#SBATCH --partition=dzagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000M
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:NVIDIAA80080GBPCIeLC:2
#SBATCH --output=turner-l30-%j.out

set -euo pipefail

TURNER_LENGTH="${TURNER_LENGTH:-30}"
if [[ "$TURNER_LENGTH" != 30 ]]; then
  echo "TURNER_LENGTH=$TURNER_LENGTH does not fit the L=30 resource class" >&2
  exit 2
fi
export TURNER_LENGTH
export TURNER_ALLOWED_LENGTHS="30"

readonly TURNER_SHARED_ROOT="/work/share/giggleliu/jiangweiqi"
if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "SLURM_SUBMIT_DIR is required" >&2
  exit 2
fi
if [[ ! -d "$SLURM_SUBMIT_DIR" ]]; then
  echo "SLURM_SUBMIT_DIR must be an existing directory" >&2
  exit 2
fi
canonical_root="$(realpath -e -- "$TURNER_SHARED_ROOT" 2>/dev/null)" || {
  echo "approved shared root is unavailable: $TURNER_SHARED_ROOT" >&2
  exit 2
}
canonical_submit_dir="$(realpath -e -- "$SLURM_SUBMIT_DIR" 2>/dev/null)" || {
  echo "cannot canonicalize SLURM_SUBMIT_DIR" >&2
  exit 2
}
case "$canonical_submit_dir" in
  "$canonical_root"/*) ;;
  *)
    echo "SLURM_SUBMIT_DIR resolves outside $TURNER_SHARED_ROOT" >&2
    exit 2
    ;;
esac
expected_runner="$canonical_submit_dir/scripts/turner2018_dzeshell_run.sh"
canonical_runner="$(realpath -e -- "$expected_runner" 2>/dev/null)" || {
  echo "Dzeshell common runner is missing" >&2
  exit 2
}
if [[ "$canonical_runner" != "$expected_runner" || ! -f "$canonical_runner" ]]; then
  echo "Dzeshell common runner must be the canonical checkout file" >&2
  exit 2
fi
source "$canonical_runner"
