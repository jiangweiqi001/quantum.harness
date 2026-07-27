#!/usr/bin/env bash
#SBATCH --job-name=turner-ed-l22-28
#SBATCH --partition=dzagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000M
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:NVIDIAA80080GBPCIeLC:1
#SBATCH --output=turner-l22-28-%j.out

set -euo pipefail

: "${TURNER_LENGTH:?set TURNER_LENGTH to 22, 24, 26, or 28}"
case "$TURNER_LENGTH" in
  22|24|26|28) ;;
  *)
    echo "TURNER_LENGTH=$TURNER_LENGTH does not fit the L=22-28 resource class" >&2
    exit 2
    ;;
esac
export TURNER_LENGTH
export TURNER_ALLOWED_LENGTHS="22|24|26|28"

: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
source "$SLURM_SUBMIT_DIR/scripts/turner2018_dzeshell_run.sh"
