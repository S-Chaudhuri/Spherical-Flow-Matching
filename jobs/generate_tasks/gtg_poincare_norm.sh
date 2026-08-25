#!/bin/bash

dims=(2 3 4 8 16 64 256 512)
seeds=(34 42 50)
curvatures=(1 2 3 5 10)

dist=0.5
std_x0=0.3
std_x1=0.3

output_file="tasks/gtg_poincare_norm.txt"
> "$output_file"

for seed in "${seeds[@]}"; do
  for dim in "${dims[@]}"; do
    for curv in "${curvatures[@]}"; do
      read mean_x0 mean_x1 <<< $(python - <<EOF
import numpy as np

dim = $dim
dist = float($dist)
curv = float($curv)

x0 = np.zeros(dim)

u = np.ones(dim)
u = u / np.linalg.norm(u)
x1 = dist * u

mean_x0 = "[" + ",".join([f"{v:.8f}" for v in x0]) + "]"
mean_x1 = "[" + ",".join([f"{v:.8f}" for v in x1]) + "]"

print(mean_x0, mean_x1)
EOF
)

      curv_tag=$(echo $curv | sed 's/\./p/g')

      echo "srun python train.py experiment=general_fm seed=${seed} hydra.run.dir=/scratch-shared/$USER/outputs/runs/gtg/poincare/norm/poi_d${dim}_c${curv_tag}_s${seed} general.manifold=poincare general.dim=${dim} general.curvature=${curv}.0 general.std_x0=${std_x0} general.std_x1=${std_x1} general.mean_x0=${mean_x0} general.mean_x1=${mean_x1} general.normalize_tangent_distributions=True" >> "$output_file"

    done
  done
done

echo "Successfully generated $(wc -l < "$output_file") tasks in $output_file"
