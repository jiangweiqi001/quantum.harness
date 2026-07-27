#!/usr/bin/env bash
#SBATCH --job-name=turner-ed-l32
#SBATCH --partition=dzagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000M
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:NVIDIAA80080GBPCIeLC:4
#SBATCH --output=turner-l32-%j.out

set -euo pipefail

TURNER_LENGTH="${TURNER_LENGTH:-32}"
if [[ "$TURNER_LENGTH" != 32 ]]; then
  echo "TURNER_LENGTH=$TURNER_LENGTH does not fit the L=32 resource class" >&2
  exit 2
fi
export TURNER_LENGTH
export TURNER_ALLOWED_LENGTHS="32"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/turner2018_dzeshell_run.sh"
