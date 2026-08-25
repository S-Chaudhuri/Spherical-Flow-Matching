#!/bin/bash

dims=(2 3 4 8 16 64 256 512)
seeds=(34 42 50)
curvatures=(1 2 3 5 10)

dist=0.5
std_x0=0.3
std_x1=0.3

output_file="tasks/gtg_sphere_norm.txt"
> "$output_file"

for seed in "${seeds[@]}"; do
  for dim in "${dims[@]}"; do
    for curv in "${curvatures[@]}"; do

      read mean_x0 mean_x1 <<< $(python - <<EOF
import numpy as np

dim = $dim
curv = float($curv)
dist = float($dist)


# north pole
# R = 1.0 / math.sqrt(curv)
mu0 = np.zeros(dim)
# mu0[0] = R

# point at geodesic distance dist from mu0
# theta = dist / R
spread = dist / np.sqrt(dim - 1)
mu1 = np.zeros(dim)
mu1[1:] = spread
# mu1[0] = R * math.cos(theta)
# mu1[1] = R * math.sin(theta)

mean_x0 = "[" + ",".join([f"{v:.7f}" for v in mu0]) + "]"
mean_x1 = "[" + ",".join([f"{v:.7f}" for v in mu1]) + "]"

print(mean_x0, mean_x1)
EOF
)

      curv_tag=$(echo $curv | sed 's/\./p/g')

      echo "srun python train.py experiment=general_fm seed=${seed} hydra.run.dir=/scratch-shared/$USER/outputs/runs/gtg/sphere/norm/sph_d${dim}_c${curv_tag}_s${seed} general.manifold=sphere general.dim=${dim} general.curvature=${curv}.0 general.std_x0=${std_x0} general.std_x1=${std_x1} general.mean_x0=${mean_x0} general.mean_x1=${mean_x1} general.normalize_tangent_distributions=True" >> "$output_file"

    done
  done
done

echo "Successfully generated $(wc -l < "$output_file") tasks in $output_file"