#!/bin/bash
#SBATCH --job-name=mog_euc
#SBATCH --time=08:00:00            # Adjust as needed
#SBATCH --partition=gpu_h100       # Standard Snellius GPU partition
#SBATCH --gpus=1                   # GPUs per task
#SBATCH --cpus-per-task=9          # Standard CPU allocation for 1 GPU on Snellius
#SBATCH --array=1-24               
#SBATCH --output=/scratch-shared/%u/output/mog/euclidean/norm_array_%A_%a.out 
#SBATCH --error=/scratch-shared/%u/logs/tch-shared/%u/logs/tch-shared/%u/logs/mog/euclidean/norm_array_%A_%a.err  

# Create necessary directories
mkdir -p /scratch-shared/$USER/logs/mog/eclidean 
mkdir -p /scratch-shared/$USER/output/mog/eclidean 
mkdir -p /scratch-shared/$USER/failed_tasks/mog/eclidean

TASKS_FILE="tasks/mog_euclidean.txt"

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
    cp "/scratch-shared/$USER/output/mog/eclidean/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" /scratch-shared/$USER/failed_tasks/mog/eclidean/
    cp "/scratch-shared/$USER/logs/mog/eclidean/norm_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" /scratch-shared/$USER/failed_tasks/mog/eclidean/
    
    echo "Logs copied to failed_tasks/mog/eclidean/ directory."
fi

exit $EXIT_CODE