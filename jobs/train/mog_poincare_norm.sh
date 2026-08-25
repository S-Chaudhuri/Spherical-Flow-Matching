#!/bin/bash
#SBATCH --job-name=mog_poi_norm
#SBATCH --time=00:30:00            # Adjust as needed
#SBATCH --partition=gpu_a100       # Standard Snellius GPU partition
#SBATCH --gpus=1                   # GPUs per task
#SBATCH --cpus-per-task=9          # Standard CPU allocation for 1 GPU on Snellius
#SBATCH --ntasks-per-node=2
#SBATCH --array=1               
#SBATCH --output=slurm_output/mog/poincare/norm_array_%A_%a.out 
#SBATCH --error=slurm_output/mog/poincare/norm_array_%A_%a.err  

PROJECT_ROOT="$HOME/Spherical-Flow-Matching"
TASKS_FILE="$PROJECT_ROOT/tasks/mog_poincare_norm.txt"

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

# 2. Correctly create nested directories
export WANDB_CACHE_DIR="/scratch-shared/$USER/wandb-cache"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$PROJECT_ROOT/metrics_links/mog/poincare"

# Extract the command corresponding to the current task ID
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")

# Dynamically extract the Hydra run directory from the command string
RAW_RUN_DIR=$(echo "$COMMAND" | grep -oP 'hydra.run.dir=\K\S+')
RUN_DIR=$(eval echo "$RAW_RUN_DIR")

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Command: $COMMAND"
echo "--------------------------------------------------------"

cd "$PROJECT_ROOT/riemannian-fm"

# Execute the command. stdout and stderr are automatically routed by Slurm
eval "$COMMAND"
EXIT_CODE=$?

# If the exit code is 0, the job succeeded. Create the symlink.
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully. Setting up metrics symlink..."
    
    LOCAL_SYMLINK_DIR="$PROJECT_ROOT/metrics_links/mog/poincare"
    
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
    
    # Copy both standard output and error files to the failed_tasks directory
    cp "$PROJECT_ROOT/slurm_output/mog/poincare/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$PROJECT_ROOT/failed_tasks/mog/poincare/"
    cp "$PROJECT_ROOT/slurm_output/mog/poincare/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$PROJECT_ROOT/failed_tasks/mog/poincare/"
    
    echo "Logs copied to failed_tasks/mog/poincare/ directory."
fi

exit $EXIT_CODE