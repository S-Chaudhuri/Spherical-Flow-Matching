# Geometric Flow Matching: Visualization Guide

This guide explains how to use the `visualize_run.py` script to evaluate whether your Flow Matching models are learning correctly. 

The script reads the `.pt` data files downloaded from Weights & Biases (W&B) and generates intuitive, geometry-aware plots showing how mass flows from your source distribution to your target distribution.

---

## 🛠 Step 1: Prerequisites

Before running the visualization script, ensure you have the required Python libraries installed. Run this in your terminal:

```bash
pip install torch numpy matplotlib
```

--- 

## 🚀 Step 2: How to Use the Script
The script is run from your command line (terminal). It requires you to point it to a specific `.pt` file that you downloaded using the artifact downloader.

### Arguments
- `--file` (or `-f`): **Required.** The path to the `.pt` file you want to analyze.
- `--save` (or `-s`): **Optional.** Add this flag to save the plot as an image file instead of opening an interactive pop-up window. (Crucial if you are working on a remote server without a display).
- `--out` (or `-o`): **Optional.** If using `--save`, this specifies the name of the output image file. Default is `run_analysis.png`.

### Example 1: View interactively on your laptop
If you are running this on your personal computer and want a pop-up window to inspect the plot:

```bash
python visualize_run.py -f ./artifacts/v20/final_fixed_eval_outputs.pt
```

### Example 2: Save to an image file (Remote Servers)
If you are SSH'd into a remote server, pop-up windows will crash. Tell the script to save it as a PNG so you can download and view it:

```bash
python visualize_run.py -f ./artifacts/v20/final_fixed_eval_outputs.pt --save --out v20_analysis.png
```

--- 

## 🧠 Step 3: How to Interpret the Plots

The script automatically detects the geometry (Euclidean, Poincaré, or Spherical) and draws the correct boundary shapes. It generates a three-panel plot. Here is exactly what you need to look for to verify if your experiment was successful:

### Panel 1: Distribution Matching
This panel shows the start and end states of your data points.
- **Gray dots (`Source`):** Where the points started (e.g., a standard Gaussian).
- **Blue dots (`True Target`):** Where the points are supposed to go.
- **Red dots (`Generated`):** Where your trained model actually moved the points.
- **✅ Success:** The Red dots should sit almost perfectly on top of the Blue dots.
- **❌ Failure:** The Red dots are scattered randomly, collapsed into a single tight cluster, or haven't moved far from the Gray dots.

### Panel 2: Geodesic Trajectories
This panel shows the true mathematical paths (geodesics) connecting the source to the target.
- **Gray lines:** These are the optimal routes that the points travel across the curvature of the manifold. It helps you visualize how space is bending (e.g., paths curving around the Poincaré disk).

### Panel 3: Field Alignment (The Most Important Metric)
Flow Matching trains a neural network to predict a "vector field" (velocities) that pushes the points along the trajectories.
- **Blue Arrows (`u_t`):** The true, optimal mathematical velocity at a specific point in time.
- **Red Arrows (`vtheta`):** The velocity your neural network predicted.
- **✅ Success:** Every Red arrow should point in the exact same direction and be the exact same length as the Blue arrow underneath it.
- **❌ Failure:** The Red arrows point in random directions, or are vastly different in size compared to the Blue arrows. This indicates the model's loss did not converge.

---

## 🌐 Supported Geometries
The script reads the metadata inside your `.pt` file and will automatically format the plots for:
1. **Euclidean Space:** Standard flat 2D plotting.
2. **Poincaré Disk (Hyperbolic):** Plots inside the unit circle, clamping axes to show the boundary of negative curvature.
3. **1D Sphere (Circle):** If dimension=2 and manifold=Sphere, it plots a 2D circle with radius $R = 1/\sqrt{K}$.
4. **2D Sphere (Surface):** If dimension=3 and manifold=Sphere, it plots a 3D wireframe globe with radius $R = 1/\sqrt{K}$.