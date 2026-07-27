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

: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
source "$SLURM_SUBMIT_DIR/scripts/turner2018_dzeshell_run.sh"
