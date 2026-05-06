# 🌌 Manifold Flow Matching: MoG Configuration Generator

Welcome to the Mixture of Gaussians (MoG) Configuration Generator. 

Defining complex distributions on curved manifolds (like the Poincaré disk or Spheres) using YAML files is notoriously error-prone. You have to manually calculate coordinates, ensure tangent-space mappings are correct, and manage massive nested lists. 

The `generate_mog_config.py` script automates this. It allows you to build complex, multi-level concentric rings of Gaussians directly from the command line, automatically calculating the exact mathematical coordinates on the tangent plane of the origin ($T_0 M$) based on your desired geodesic distances.

---

## 🧠 The Math: How We Handle Curvature

Before using the script, it is important to understand how means are placed. 

Regardless of whether you are working in Euclidean, Spherical ($K>0$), or Hyperbolic ($K<0$) space, **we define our Gaussians on the tangent plane at the origin ($T_0 M$).** The tangent plane is a perfectly flat, Euclidean space ($\mathbb{R}^n$). 

Therefore, when you specify a `radius` in this script, you are specifying the **exact geodesic distance** across the manifold. 
1. The script places the mean on the flat tangent plane using standard polar coordinates.
2. During training, the mathematical Exponential Map ($\text{Exp}_0(v)$) automatically "wraps" these flat coordinates onto the curved manifold perfectly. 

---

## 🚀 Quick Start

Run the script from your terminal. It outputs a fully populated `.yaml` file directly into the `configs/experiment/` directory.

```bash
python generate_mog_config.py --filename my_mog.yaml --manifold poincare --counts 3 --radii 2.0 --stds 0.1 --weights 1.0
```

---

## 🏗️ The Core Concept: MoG Topology (Levels)
The script builds Gaussians in "levels" (or concentric rings). The core topological arguments: `--counts`, `--radii`, `--stds`, and `--weights`, operate as **parallel lists**.

If you want an inner ring and an outer ring, you pass two values to each of these arguments.
- `--counts` (List of Ints): How many Gaussians exist at this level? The script will automatically distribute them evenly (e.g., 3 Gaussians = 120° apart).
- `--radii` (List of Floats): The geodesic distance from the origin for this level.
- `--stds` (List of Strings): The spread of the Gaussians.
    - **Isotropic:** Pass a single number (e.g., `0.1`). It will apply to all dimensions.
    - **Anisotropic:** Pass a comma-separated string matching your dimensionality (e.g., `"0.5, 0.1, 0.0"` for a 3D space).
- `--weights` (List of Floats): The importance of the Gaussians at this level. The script automatically normalizes all weights across all levels to sum to `1.0`.

### Example: A 2-Level Setup
```bash
python generate_mog_config.py \
  --counts 3 6 \
  --radii 1.5 3.0 \
  --stds 0.1 0.2 \
  --weights 1.0 0.5
```

**What this does:** Creates an inner ring of 3 tight Gaussians (std=0.1) at a distance of 1.5. Creates an outer ring of 6 wider Gaussians (std=0.2) at a distance of 3.0. The inner Gaussians are weighted twice as heavily as the outer ones.

---

## 🎯 Advanced Targeting: The --overrides Argument
Some experiments may require a highly asymmetric distribution where one specific Gaussian acts differently than the rest of its ring. You can target individual Gaussians using a JSON dictionary passed to `--overrides`.

**Naming Convention:** Gaussians are named `L{level_index}_G{gaussian_index}`.
- `L0_G0` is the first Gaussian in the first ring.
- `L1_G2` is the third Gaussian in the second ring.

**Usage:** Pass a JSON string defining the new `weight` and/or `std` (as a full list).

### Example: Overriding a specific Gaussian
```bash
python generate_mog_config.py \
  --counts 4 \
  --radii 2.0 \
  --stds 0.2 \
  --weights 1.0 \
  --overrides '{"L0_G0": {"weight": 10.0, "std": [1.0, 0.1, 0.0]}}'
```

**What this does:** Creates 4 Gaussians in a ring. However, the first Gaussian (`L0_G0`) is given a massive weight relative to the others and is heavily stretched along the X-axis (std=[1.0, 0.1, 0.0]).

---

## 🎛️ Full Argument Reference
### File & General Settings
| Argument | Default | Description | 
| --- | --- | --- | 
| `--filename` | `mog_experiment.yaml` | Output name. Saved in `configs/experiment/` | 
| `--manifold` | `poincare` | Geometry type (`poincare`, `sphere`, `euclidean`). | 
| `--curvature` | `1.0` | Mathematical curvature $K$. | 
| `--dim` | `3` | Dimensionality of the data. | 
| `--x0_dist` | `gaussian` | The base/source distribution. | 
| `--n_samples` | `20000` | Number of points to sample. | 
| `--std_x0` | `0.7` | Spread of the source distribution. | 

### Boolean Flags (Toggles)
We use a standard flag system for booleans.
- **To turn ON a feature (defaults to False):** Use the flag directly (e.g., `--eval_projx`).
- **To turn OFF a feature (defaults to True):** Use the `no_` prefix (e.g., `--no_visualize`, `--no_metric_normalize`).

| Flag | Action |
| --- | --- |
| `--no_metric_normalize` | Disables metric normalization in the model. | 
| `--eval_projx` | Enables projection during evaluation steps. | 
| `--local_coords` | Forces the model to operate in local tangent coordinates. | 
| `--normalize_loglik` | Normalizes the computed log-likelihood. | 
| `--no_visualize` | Disables automatic W&B visualization plotting. | 

### Model & Optimizer Settings
You can override any baseline architectural parameters directly.
- `--d_model` (512), `--num_layers` (5), `--actfn` (swish)
- `--num_iterations` (10000), `--batch_size` (512), `--lr` (1e-4)
- `--eval_t_values`: Pass a list of floats (e.g., `--eval_t_values 0.0 0.5 1.0`) to dictate exactly which timesteps are integrated and plotted.

---

## 📖 Cookbook: Practical Examples
### 1. The "Bullseye" (1 Center, 1 Outer Ring)
To put a Gaussian exactly at the origin, use a level with a count of 1 and a radius of 0.0.

```bash
python generate_mog_config.py --filename bullseye.yaml --manifold poincare --counts 1 5 --radii 0.0 2.5 --stds 0.5 0.1 --weights 2.0 1.0
```

### 2. The 2D Euclidean Test
If you want to test standard flat flow matching in 2D to ensure your model works before adding curvature.
```bash
python generate_mog_config.py --filename flat_test.yaml --manifold euclidean --dim 2 --counts 4 --radii 5.0 --stds 0.5 --weights 1.0
```

### 3. Anisotropic Spherical Poles
Placing two stretched Gaussians on opposite sides of a sphere (distance $\pi/2$ from the tangent origin).
```bash
python generate_mog_config.py --filename sphere_poles.yaml --manifold sphere --curvature 1.0 --counts 2 --radii 1.5707 --stds "0.8,0.1,0.1" --weights 1.0
```