#!/bin/bash
#SBATCH --job-name=sgr_sph_norm
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=1-270
#SBATCH --output=/slurm_output/sgr/sphere/norm_array_%A_%a.out 
#SBATCH --error=/slurm_output/sgr/sphere/norm_array_%A_%a.err  

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
mkdir -p slurm_output/sgr/sphere 
mkdir -p failed_tasks/sgr/sphere

TASKS_FILE="tasks/sgr_sphere_norm.txt"

# 3. Extract the command
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")

# Dynamically extract the Hydra run directory from the command string
RUN_DIR=$(echo "$COMMAND" | grep -oP 'hydra.run.dir=\K\S+')

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Command: $COMMAND"
echo "--------------------------------------------------------"

# 4. Execute the command
eval "$COMMAND"
EXIT_CODE=$?

# If the exit code is 0, the job succeeded. Create the symlink.
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully. Setting up metrics symlink..."
    
    LOCAL_SYMLINK_DIR="metrics_links/sgr/sphere"
    mkdir -p "$LOCAL_SYMLINK_DIR"
    
    # Extract just the run name (e.g., euc_d2_s34) to cleanly name the local link
    RUN_NAME=$(basename "$RUN_DIR")
    
    if [ -f "$RUN_DIR/metrics.json" ]; then
        ln -sf "$RUN_DIR/metrics.json" "$LOCAL_SYMLINK_DIR/${RUN_NAME}_metrics.json"
        echo "Symlink created: $LOCAL_SYMLINK_DIR/${RUN_NAME}_metrics.json"
    else
        echo "Warning: metrics.json not found in $RUN_DIR"
    fi

# If the exit code is not 0, the job failed.
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with exit code $EXIT_CODE."
    
    cp "/slurm_output/sgr/sphere/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" /failed_tasks/sgr/sphere/
    cp "/slurm_output/sgr/sphere/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" /failed_tasks/sgr/sphere/
    
    echo "Logs copied to failed_tasks/sgr/sphere/ directory."
fi

exit $EXIT_CODE