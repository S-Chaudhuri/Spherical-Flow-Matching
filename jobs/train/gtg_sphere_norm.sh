#!/bin/bash
#SBATCH --job-name=gtg_sph_norm
#SBATCH --time=08:00:00            # Adjust as needed
#SBATCH --partition=gpu_h100       # Standard Snellius GPU partition
#SBATCH --gpus=1                   # GPUs per task
#SBATCH --cpus-per-task=9          # Standard CPU allocation for 1 GPU on Snellius
#SBATCH --array=1-120               
#SBATCH --output=/scratch-shared/%u/output/gtg/sphere/norm_array_%A_%a.out 
#SBATCH --error=/scratch-shared/%u/logs/gtg/sphere/norm_array_%A_%a.err  

# Create necessary directories
mkdir -p /scratch-shared/$USER/logs/gtg/sphere 
mkdir -p /scratch-shared/$USER/output/gtg/sphere 
mkdir -p /scratch-shared/$USER/failed_tasks/gtg/sphere

TASKS_FILE="tasks/gtg_sphere_norm.txt"

# Extract the command corresponding to the current task ID
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Command: $COMMAND"
echo "--------------------------------------------------------"

# Execute the command. stdout and stderr are automatically routed by Slurm
eval "$COMMAND"
EXIT_CODE=$?

# If the exit code is not 0, the job failed
if [ $EXIT_CODE -ne 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID failed with exit code $EXIT_CODE."
    
    # Copy both standard output and error files to the failed_tasks directory
    cp "/scratch-shared/$USER/output/gtg/sphere/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" /scratch-shared/$USER/failed_tasks/gtg/sphere/
    cp "/scratch-shared/$USER/logs/gtg/sphere/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" /scratch-shared/$USER/failed_tasks/gtg/sphere/

    echo "Logs copied to failed_tasks/gtg/sphere/ directory."
fi

exit $EXIT_CODE