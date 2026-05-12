import argparse
import json
import math
import os

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
    general_group.add_argument("--origin", type=float, nargs='+', default=None, help="Origin point on the manifold (default: null)")

    # --- Explicit MoG Arguments ---
    mog_group = parser.add_argument_group("Explicit Mixture of Gaussians Parameters")
    mog_group.add_argument("--radii", type=float, nargs='+', required=True, help="List of floats: Geodesic distances for each Gaussian from the origin.")
    mog_group.add_argument("--angles", type=float, nargs='+', required=True, help="List of floats: Angles in degrees relative to the horizontal axis for each Gaussian.")
    mog_group.add_argument("--stds", type=str, nargs='+', required=True, help="List of strings: Standard deviations per Gaussian. Use '0.1' for isotropic or '0.1,0.2,0.1' for anisotropic.")
    mog_group.add_argument("--weights", type=float, nargs='+', required=True, help="List of floats: Base importance weight for each Gaussian.")
    mog_group.add_argument("--overrides", type=str, default="{}", help="JSON string to override specific Gaussians. Format: '{\"G1\": {\"weight\": 2.0, \"std\": [0.5, 0.5, 0.0]}}'")

    # --- Metrics Used (Toggles) ---
    metrics_group = parser.add_argument_group("Metrics Used Settings")
    metrics_group.add_argument("--no_sinkhorn_knopp", action="store_false", dest="sinkhorn_knopp", help="Disable Sinkhorn-Knopp metric")
    metrics_group.add_argument("--no_mmd", action="store_false", dest="mmd", help="Disable MMD metric")
    metrics_group.add_argument("--no_epsilon_coverage", action="store_false", dest="epsilon_coverage", help="Disable Epsilon Coverage metric")
    metrics_group.add_argument("--no_epsilon_precision", action="store_false", dest="epsilon_precision", help="Disable Epsilon Precision metric")
    metrics_group.add_argument("--no_frechet_variance", action="store_false", dest="frechet_variance", help="Disable Frechet Variance metric")
    metrics_group.add_argument("--no_dispersion", action="store_false", dest="dispersion", help="Disable Dispersion metric")
    metrics_group.add_argument("--no_radial", action="store_false", dest="radial", help="Disable Radial metric")
    metrics_group.add_argument("--no_stability", action="store_false", dest="stability", help="Disable Stability metric")
    metrics_group.add_argument("--no_rfm", action="store_false", dest="rfm", help="Disable RFM metric")
    metrics_group.add_argument("--cross_curvature", action="store_true", dest="cross_curvature", help="Enable Cross-Curvature metric")

    # --- Metrics Parameters ---
    metrics_param_group = parser.add_argument_group("Metrics Parameters")
    metrics_param_group.add_argument("--sinkhorn_blur", type=float, default=0.05, help="Sinkhorn blur parameter")
    metrics_param_group.add_argument("--coverage_eps_multiplier", type=float, default=1.0, help="Coverage epsilon multiplier")
    metrics_param_group.add_argument("--save_densities", action="store_true", help="Enable saving densities (sets save_densities: True)")

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
    # fmt: on


def parse_std_string(std_str, dim):
    """Converts a CLI string like '0.1' or '0.1,0.2' into a list of length 'dim'."""

    parts = [float(x) for x in std_str.split(",")]
    if len(parts) == 1:
        return parts * dim  # Broadcast isotropic to all dims

    elif len(parts) == dim:
        return parts

    else:
        raise ValueError(f"Standard deviation '{std_str}' must have 1 or {dim} elements.")


def generate_mog_parameters(args):
    """Calculates the explicit means, stds, and weights for the MoG in the tangent space."""

    all_means = []
    all_stds = []
    all_weights = []

    overrides = json.loads(args.overrides)

    # Validate that all explicit MoG lists are of the same length
    num_gaussians = len(args.radii)
    if not (
        len(args.angles) == num_gaussians
        and len(args.stds) == num_gaussians
        and len(args.weights) == num_gaussians
    ):
        msg = "The number of arguments provided to --radii, --angles, --stds, and --weights must be identical."
        raise ValueError(msg)

    for g_idx in range(num_gaussians):
        identifier = f"G{g_idx}"

        radius = args.radii[g_idx]
        angle_deg = args.angles[g_idx]
        std_str = args.stds[g_idx]
        weight = args.weights[g_idx]

        base_std = parse_std_string(std_str, args.dim)

        # 1. Calculate Mean in Tangent Space
        # Convert user-provided degrees to radians
        angle_rad = math.radians(angle_deg)
        x = radius * math.cos(angle_rad)
        y = radius * math.sin(angle_rad)

        # Pad with zeros if dimensionality > 2. This embeds the points in the 2D xy-plane.
        if args.dim >= 2:
            mean = [round(x, 4), round(y, 4)] + [0.0] * (args.dim - 2)

        else:
            mean = [round(x, 4)]  # Edge case for 1D

        # 2. Check for Specific Overrides
        std = base_std
        if identifier in overrides:
            if "weight" in overrides[identifier]:
                weight = overrides[identifier]["weight"]

            if "std" in overrides[identifier]:
                std = overrides[identifier]["std"]

        all_means.append(mean)
        all_stds.append(std)
        all_weights.append(weight)

    # Normalize weights to sum to 1.0 (standard requirement for a valid MoG)
    total_weight = sum(all_weights)
    if total_weight == 0:
        raise ValueError("Total weight of Gaussians cannot be zero.")

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
            "x1_dist": "MoG",  # Forcing this to Mixture of Gaussians
            "n_samples": args.n_samples,
            "std_x0": args.std_x0,
            "mean_x0": args.mean_x0,
            "std_x1": stds,
            "mean_x1": means,
            "weights": weights,
        },
        "metrics_used": {
            "sinkhorn_knopp": args.sinkhorn_knopp,
            "mmd": args.mmd,
            "epsilon_coverage": args.epsilon_coverage,
            "epsilon_precision": args.epsilon_precision,
            "frechet_variance": args.frechet_variance,
            "dispersion": args.dispersion,
            "radial": args.radial,
            "stability": args.stability,
            "rfm": args.rfm,
            "cross_curvature": args.cross_curvature,
        },
        "metrics_param": {
            "sinkhorn_blur": args.sinkhorn_blur,
            "coverage_eps_multiplier": args.coverage_eps_multiplier,
            "save_densities": args.save_densities,
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
    sections_to_space = ["general:", "metrics_used:model:", "optim:", "val_every:"]
    comments_to_add = [
        "# --- Explicit MoG Arguments ---",
        "# --- Metrics Parameters ---",
        "# --- Flow Matching Model Settings ---",
        "# --- Optimizer Settings ---",
        "# --- Evaluation & Logging Settings ---",
    ]

    for comment, section in zip(comments_to_add, sections_to_space):
        yaml_str = yaml_str.replace(f"\n{section}", f"\n\n{comment}\n{section}")

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
