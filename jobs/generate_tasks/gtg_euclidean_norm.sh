#!/bin/bash

dims=(2 3 4 8 16 64 256 512)
seeds=(34 42 50)

dist=0.5
std_x0=0.3
std_x1=0.3

output_file="tasks/gtg_euclidean.txt"

for seed in "${seeds[@]}"; do
  for dim in "${dims[@]}"; do

    read mean_x0 mean_x1 <<< $(python - <<EOF
import numpy as np

dim = $dim
dist = $dist

u = np.ones(dim)
u = u / np.linalg.norm(u)

x0 = np.zeros(dim)
x1 = dist * u

mean_x0 = "[" + ",".join([f"{v:.8f}" for v in x0]) + "]"
mean_x1 = "[" + ",".join([f"{v:.8f}" for v in x1]) + "]"

print(mean_x0, mean_x1)
EOF
)

    echo "srun python train.py experiment=general_fm seed=${seed} hydra.run.dir=/scratch-shared/$USER/outputs/runs/gtg/euclidean/general/euc_d${dim}_s${seed} general.manifold=euclidean general.dim=${dim} general.curvature=0.0 general.std_x0=${std_x0} general.std_x1=${std_x1} general.mean_x0=${mean_x0} general.mean_x1=${mean_x1} general.normalize_tangent_distributions=True" >> "$output_file"
  
  done
done

echo "Successfully generated $(wc -l < "$output_file") tasks in $output_file"