#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=test_new_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:20:00
#SBATCH --output=output/slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate manifm

cd "$HOME/Spherical-Flow-Matching/riemannian-fm"

srun "$CONDA_PREFIX/bin/python" train.py experiment=general_fm \
  seed=34 \
  hydra.run.dir=outputs/runs/gtg/euclidean/test/euc_d2_dist0p5_s34 \
  general.manifold=euclidean \
  general.dim=2 \
  general.curvature=0.0 \
  general.std_x0=0.3 \
  general.std_x1=0.3 \
  "general.mean_x0=[0.00000000,0.00000000]" \
  "general.mean_x1=[0.35355339,0.35355339]" \
  general.normalize_tangent_distributions=False