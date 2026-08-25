#!/bin/bash

dims=(2 3 16 64 256 512)
seeds=(34 42 50)
curvatures=(1 2 3 5 10)
modes=(2 3 5)
balances=("balanced" "unbalanced")

dist=0.5
std_x0=0.3
std_x1=0.3

output_file="tasks/mog_poincare_norm.txt"
> "$output_file"

for seed in "${seeds[@]}"; do
  for dim in "${dims[@]}"; do
    for curv in "${curvatures[@]}"; do
      for n in "${modes[@]}"; do
        for balance in "${balances[@]}"; do

          read means_x0 means_x1 weights stds <<< $(python - <<EOF
import numpy as np

n = $n
dim = $dim
dist = float($dist)
balance = "$balance"

vectors = []
if dim == 1:
    vectors = [[dist if i % 2 == 0 else -dist] for i in range(n)]

elif n == 2:
    vectors = [[dist, 0.0], [-dist, 0.0]]

elif n == 3:
    vectors = [[dist, 0.0], [-0.5*dist, 0.5*np.sqrt(3)*dist], [-0.5*dist, -0.5*np.sqrt(3)*dist]]

elif n == 5:
    if dim == 2:
        vectors = [[dist*np.cos(2*np.pi*k/5), dist*np.sin(2*np.pi*k/5)] for k in range(5)]
    
    else:
        vectors = [[0.0, 0.0, dist], [0.0, 0.0, -dist], [dist, 0.0, 0.0], 
                   [-0.5*dist, 0.5*np.sqrt(3)*dist, 0.0], [-0.5*dist, -0.5*np.sqrt(3)*dist, 0.0]]

final_vectors = []
for v in vectors:
    v_padded = list(v)
    while len(v_padded) < dim:
        v_padded.append(0.0)
    final_vectors.append(v_padded[:dim])

if balance == "balanced":
    w = [1.0/n] * n
else:
    if n == 2: w = [1/3, 2/3]
    elif n == 3: w = [1/4, 1/4, 2/4]
    elif n == 5: w = [1/8, 2/8, 3/8, 1/8, 1/8]

x0 = np.zeros(dim)

means_x0_str = "[" + ",".join([f"{v:.7f}" for v in x0]) + "]"
means_x1_str = "[" + ",".join(["[" + ",".join([f"{val:.7f}" for val in vec]) + "]" for vec in final_vectors]) + "]"
weights_str = "[" + ",".join([f"{val:.4f}" for val in w]) + "]"
stds_str = "[" + ",".join([f"{float($std_x1)}"] * n) + "]"

print(means_x0_str, means_x1_str, weights_str, stds_str)
EOF
)
          curv_tag=$(echo $curv | sed 's/\./p/g')

          echo "srun python train.py experiment=general_fm seed=${seed} hydra.run.dir=/scratch-shared/$USER/outputs/runs/mog/poincare/norm/poi_d${dim}_c${curv_tag}_n${n}_${balance}_s${seed} general.manifold=poincare general.dim=${dim} general.curvature=${curv}.0 general.x1_dist=mog general.std_x0=${std_x0} general.std_x1=${stds} general.weights=${weights} general.mean_x0=${means_x0} general.mean_x1=${means_x1} general.normalize_tangent_distributions=True" >> "$output_file"

        done
      done
    done
  done
done

echo "Successfully generated $(wc -l < "$output_file") tasks in $output_file"