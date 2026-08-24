#!/bin/bash
#SBATCH --job-name=gtg_poi_norm
#SBATCH --time=08:00:00            # Adjust as needed
#SBATCH --partition=gpu_h100       # Standard Snellius GPU partition
#SBATCH --gpus=1                   # GPUs per task
#SBATCH --cpus-per-task=9          # Standard CPU allocation for 1 GPU on Snellius
#SBATCH --array=1-120               
#SBATCH --output=/slurm_output/gtg/poincare/norm_array_%A_%a.out 
#SBATCH --error=/slurm_output/gtg/poincare/norm_array_%A_%a.err  

# Create necessary directories
mkdir -p slurm_output/gtg/poincare 
mkdir -p failed_tasks/gtg/poincare

TASKS_FILE="tasks/gtg_poincare_norm.txt"

# Extract the command corresponding to the current task ID
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")

# Dynamically extract the Hydra run directory from the command string
RUN_DIR=$(echo "$COMMAND" | grep -oP 'hydra.run.dir=\K\S+')

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Command: $COMMAND"
echo "--------------------------------------------------------"

# Execute the command. stdout and stderr are automatically routed by Slurm
eval "$COMMAND"
EXIT_CODE=$?

# If the exit code is 0, the job succeeded. Create the symlink.
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully. Setting up metrics symlink..."
    
    LOCAL_SYMLINK_DIR="metrics_links/gtg/poincare"
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
    
    # Copy both standard output and error files to the failed_tasks directory
    cp "/slurm_output/gtg/poincare/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" /failed_tasks/gtg/poincare/
    cp "/slurm_output/gtg/poincare/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" /failed_tasks/gtg/poincare/
    
    echo "Logs copied to failed_tasks/gtg/poincare/ directory."
fi

exit $EXIT_CODE