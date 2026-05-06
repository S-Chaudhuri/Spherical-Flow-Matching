import argparse
import json
import math
import os
import random

import yaml


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate YAML configs for Manifold FM MoG experiments.")

    # --- File & General Settings ---
    general_group = parser.add_argument_group("General & File settings")
    general_group.add_argument("--filename", type=str, default="mog_experiment.yaml", help="Name of the output YAML file")
    general_group.add_argument("--manifold", type=str, default="poincare", help="Manifold type (poincare, sphere, euclidean)")
    general_group.add_argument("--curvature", type=float, default=1.0, help="Curvature of the manifold")
    general_group.add_argument("--dim", type=int, default=3, help="Dimensionality of the data")
    general_group.add_argument("--x0_dist", type=str, default="gaussian", help="Source distribution")
    general_group.add_argument("--n_samples", type=int, default=20000, help="Number of samples")
    general_group.add_argument("--std_x0", type=float, default=0.7, help="Standard deviation of x0")
    general_group.add_argument("--mean_x0", type=float, nargs='+', default=[0.0, 0.0, 0.0], help="Mean of x0")

    # --- MoG Topology Arguments ---
    mog_group = parser.add_argument_group("Mixture of Gaussians Topology")
    mog_group.add_argument("--counts", type=int, nargs='+', required=True, help="List of integers: Number of Gaussians at each level (e.g., --counts 3 2)")
    mog_group.add_argument("--radii", type=float, nargs='+', required=True, help="List of floats: Geodesic distance (radius) for each level (e.g., --radii 1.5 3.0)")
    mog_group.add_argument("--stds", type=str, nargs='+', required=True, help="List of strings: Standard deviations per level. Use '0.1' for isotropic or '0.1,0.2,0.1' for anisotropic.")
    mog_group.add_argument("--weights", type=float, nargs='+', required=True, help="List of floats: Base importance weight for the Gaussians at each level.")
    mog_group.add_argument("--overrides", type=str, default="{}", help="JSON string to override specific Gaussians. Format: '{\"L0_G1\": {\"weight\": 2.0, \"std\": [0.5, 0.5, 0.0]}}'")

    # --- Model Settings ---
    model_group = parser.add_argument_group("Model settings")
    model_group.add_argument("--d_model", type=int, default=512, help="Model hidden dimension")
    model_group.add_argument("--num_layers", type=int, default=5, help="Number of layers")
    model_group.add_argument("--actfn", type=str, default="swish", help="Activation function")
    model_group.add_argument("--fourier", type=int, default=None, help="Fourier features dimension (default: null/None)")
    model_group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for ODE solver")
    model_group.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for ODE solver")
    model_group.add_argument("--no_metric_normalize", action="store_false", dest="metric_normalize", help="Disable metric normalization (sets metric_normalize: False)")

    # --- Optimizer Settings ---
    optim_group = parser.add_argument_group("Optimizer settings")
    optim_group.add_argument("--num_iterations", type=int, default=10000, help="Total training iterations")
    optim_group.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    optim_group.add_argument("--val_batch_size", type=int, default=2000, help="Validation batch size")
    optim_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # --- Evaluation & Logging Settings ---
    eval_group = parser.add_argument_group("Evaluation & Logging settings")
    eval_group.add_argument("--val_every", type=int, default=500, help="Frequency of validation steps")
    eval_group.add_argument("--div_mode", type=str, default="rademacher", help="Divergence estimation mode")
    eval_group.add_argument("--eval_projx", action="store_true", help="Enable projection during evaluation (sets eval_projx: True)")
    eval_group.add_argument("--local_coords", action="store_true", help="Use local coordinates (sets local_coords: True)")
    eval_group.add_argument("--normalize_loglik", action="store_true", help="Normalize log-likelihood (sets normalize_loglik: True)")
    eval_group.add_argument("--no_visualize", action="store_false", dest="visualize", help="Disable visualization logging (sets visualize: False)")
    eval_group.add_argument("--eval_n_pairs", type=int, default=100, help="Number of fixed evaluation pairs")
    eval_group.add_argument("--eval_t_values", type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0], help="List of ODE integration timesteps to evaluate/log")

    return parser.parse_args()
    # fmt: off


def parse_std_string(std_str, dim):
    """Converts a CLI string like '0.1' or '0.1, 0.2' into a list of length 'dim'."""

    parts = [float(x) for x in std_str.split(",")]

    if len(parts) == 1:
        return parts * dim  # Broadcast isotropic to all dims

    elif len(parts) == dim:
        return parts

    else:
        raise ValueError(f"Standard deviation '{std_str}' must have 1 or {dim} elements.")


def generate_mog_parameters(args):
    """Calculates the means, stds, and weights for the MoG based on tangent space geometry."""

    all_means = []
    all_stds = []
    all_weights = []

    overrides = json.loads(args.overrides)

    if not (len(args.counts) == len(args.radii) == len(args.stds) == len(args.weights)):
        error_msg = "The number of arguments provided to --counts, --radii, --stds, and --weights must be identical."
        raise ValueError(error_msg)

    for level_idx, (count, radius, std_str, level_weight) in enumerate(
        zip(args.counts, args.radii, args.stds, args.weights)
    ):
        # Initialisation of the first Gaussian of this level
        phase = 0
        angle_step = (2 * math.pi) / count

        base_std = parse_std_string(std_str, args.dim)

        for g_idx in range(count):
            identifier = f"L{level_idx}_G{g_idx}"

            # 1. Calculate Mean (assuming points are distributed on the 2D xy-plane of the tangent space)
            angle = phase + (g_idx * angle_step)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            # Pad with zeros if dimensionality is > 2
            mean = [round(x, 4), round(y, 4)] + [0.0] * (args.dim - 2)

            # 2. Check for Specific Overrides
            std = base_std
            weight = level_weight

            if identifier in overrides:
                if "weight" in overrides[identifier]:
                    weight = overrides[identifier]["weight"]

                if "std" in overrides[identifier]:
                    std = overrides[identifier]["std"]

            all_means.append(mean)
            all_stds.append(std)
            all_weights.append(weight)

    # Normalize weights to sum to 1.0 (standard for MoG)
    total_weight = sum(all_weights)
    all_weights = [round(w / total_weight, 4) for w in all_weights]

    return all_means, all_stds, all_weights


def main():
    args = parse_args()

    # Generate the mathematically correct coordinates and parameters
    means, stds, weights = generate_mog_parameters(args)

    # Construct the configuration dictionary
    config = {
        "data": "general_fm",
        "use_wandb": True,
        "general": {
            "manifold": args.manifold,
            "curvature": args.curvature,
            "dim": args.dim,
            "x0_dist": args.x0_dist,
            "x1_dist": "mog",  # Forcing this to Mixture of Gaussians
            "n_samples": args.n_samples,
            "std_x0": args.std_x0,
            "mean_x0": args.mean_x0,
            "std_x1": stds,
            "mean_x1": means,
            "weights": weights,
        },
        "model": {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "actfn": args.actfn,
            "fourier": args.fourier,
            "atol": args.atol,
            "rtol": args.rtol,
            "metric_normalize": args.metric_normalize,
        },
        "optim": {
            "num_iterations": args.num_iterations,
            "batch_size": args.batch_size,
            "val_batch_size": args.val_batch_size,
            "lr": args.lr,
        },
        "val_every": args.val_every,
        "div_mode": args.div_mode,
        "eval_projx": args.eval_projx,
        "local_coords": args.local_coords,
        "normalize_loglik": args.normalize_loglik,
        "visualize": args.visualize,
        "eval_n_pairs": args.eval_n_pairs,
        "eval_t_values": args.eval_t_values,
    }

    # 1. Dump the dictionary to a string instead of directly to a file
    yaml.Dumper.ignore_aliases = lambda *args: True
    yaml_str = yaml.dump(config, sort_keys=False, default_flow_style=None)

    # 2. Inject blank lines before major blocks to stop it from looking crammed
    sections_to_space = ["general:", "model:", "optim:", "val_every:"]
    for section in sections_to_space:
        yaml_str = yaml_str.replace(f"\n{section}", f"\n\n{section}")

    # Save to the configs/experiment directory
    os.makedirs("configs/experiment", exist_ok=True)
    filepath = os.path.join("configs/experiment", args.filename)

    with open(filepath, "w") as f:
        # Add the required Hydra header
        f.write("# @package _global_\n")
        f.write(yaml_str)

    print(f"✅ Successfully generated MoG configuration: {filepath}")
    print(f"Total Gaussians generated: {len(means)}")
    print("Configuration snippet:")
    print(f"  Means: {means[:2]} ...")
    print(f"  Stds:  {stds[:2]} ...")
    print(f"  Weights: {weights}")


if __name__ == "__main__":
    main()
