#!/bin/bash
#SBATCH --job-name=gapsplit_L8
#SBATCH --output=gsplit_L8_%A_%a.out
#SBATCH --error=gsplit_L8_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=xhacnormalb
#SBATCH --array=0-3

# gapless-point-split L=8: sparse ED, 41×51 grid
# Array task 0 → U=0.0, 1 → U=0.2, 2 → U=0.5, 3 → U=1.0
# ~20-30 min per task

U_VALUES=(0.0 0.2 0.5 1.0)
U=${U_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=== gapless-point-split L=8  U=$U ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Task:      $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo

source /work/home/mazhuijing/ed-project/.venv/bin/activate

export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "PWD: $(pwd)"
echo "Python: $(which python3)"
echo

python3 experiments/gapless-point-split/scripts/run_scan.py \
    --config experiments/gapless-point-split/configs/coarse_L8.yaml \
    --mode auto --U "$U"

echo
echo "End: $(date)"
