#!/bin/bash

# Ambient dimensions corresponding to intrinsic dimensions 2, 3, 16, 64, 256, 512
dims=(3 4 17 65 257 513)
seeds=(34 42 50)
curvatures=(1 2 3 5 10)
radii=(0.35 0.60 0.85)

output_file="tasks/sgr_sphere_norm.txt"

# Clear the file to prevent appending to old tasks
> "$output_file"

for seed in "${seeds[@]}"; do
  for dim in "${dims[@]}"; do
    for curv in "${curvatures[@]}"; do
      for radius in "${radii[@]}"; do

        curv_tag=$(echo "${curv}" | sed 's/\./p/g')
        radius_tag=$(echo "${radius}" | sed 's/\./p/g')

        echo "srun python train.py experiment=general_fm seed=${seed} \
hydra.run.dir=/scratch-shared/$USER/outputs/runs/sgr/sphere/norm/sph_d${dim}_c${curv_tag}_r${radius_tag}_s${seed}_norm \
general.manifold=sphere \
general.dim=${dim} \
general.curvature=${curv}.0 \
general.x0_dist=gaussian \
general.x1_dist=gaussian-ring \
general.std_x0=0.3 \
general.radius_x1=${radius} \
general.std_x1=0.05 \
general.normalize_tangent_distributions=True" >> "$output_file"

      done
    done
  done
done

echo "Successfully generated $(wc -l < "$output_file") tasks in $output_file"