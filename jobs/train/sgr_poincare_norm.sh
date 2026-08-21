#!/bin/bash
#SBATCH --job-name=sgr_poi_norm
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=1-270
#SBATCH --output=/scratch-shared/%u/output/sgr/poincare/norm_array_%A_%a.out 
#SBATCH --error=/scratch-shared/%u/logs/sgr/poincare/norm_array_%A_%a.err  

set -euo pipefail

# 1. Environment loading
module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate base
conda install -y -n manifm -c conda-forge libstdcxx-ng libgcc-ng
conda activate manifm
set -u

cd "$HOME/DL2_project/Spherical-Flow-Matching/riemannian-fm"
export WANDB_CACHE_DIR="/scratch-shared/$USER/wandb-cache"

# 2. Correctly create nested directories
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p /scratch-shared/$USER/logs/sgr/poincare 
mkdir -p /scratch-shared/$USER/output/sgr/poincare 
mkdir -p /scratch-shared/$USER/failed_tasks/sgr/poincare

TASKS_FILE="tasks/sgr_poincare_norm.txt"

# 3. Extract the command
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Command: $COMMAND"
echo "--------------------------------------------------------"

# 4. Execute the command
eval "$COMMAND"
EXIT_CODE=$?

# 5. Failure logic mapped to exact paths
if [ $EXIT_CODE -ne 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID failed with exit code $EXIT_CODE."
    
    cp "/scratch-shared/$USER/output/sgr/poincare/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" /scratch-shared/$USER/failed_tasks/sgr/poincare/
    cp "/scratch-shared/$USER/logs/sgr/poincare/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" /scratch-shared/$USER/failed_tasks/sgr/poincare/
    
    echo "Logs copied to failed_tasks/sgr/poincare/ directory."
fi

exit $EXIT_CODE