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
    general_group.add_argument("--normalize_tangent_distributions", action="store_true", help="Whether to normalize the tangent distribution for curvature.")

    # --- Explicit MoG Arguments ---
    mog_group = parser.add_argument_group("Explicit Mixture of Gaussians Parameters")
    mog_group.add_argument("--radii", type=float, nargs='+', default=None, help="List of floats: Geodesic distances for each Gaussian from the origin. (Polar Mode)")
    mog_group.add_argument("--angles", type=float, nargs='+', default=None, help="List of floats: Angles in degrees relative to the horizontal axis. (Polar Mode)")
    mog_group.add_argument("--cartesian_means", type=str, default=None, help="JSON string: Direct Cartesian coordinates in the tangent space. Bypasses radii/angles. Format: '[[1.0, 0.0, 0.0], [-1.0, 0.5, 0.0]]' (Cartesian Mode)")

    mog_group.add_argument("--stds", type=str, nargs='+', required=True, help="List of strings: Standard deviations per Gaussian. Use '0.1' for isotropic or '0.1,0.2,0.1' for anisotropic.")
    mog_group.add_argument("--weights", type=float, nargs='+', required=True, help="List of floats: Base importance weight for each Gaussian.")
    mog_group.add_argument("--overrides", type=str, default="{}", help="JSON string to override specific Gaussians. Format: '{\"G1\": {\"weight\": 2.0, \"std\": [0.5, 0.5, 0.0]}}'")

    # --- Metrics Used (Toggles) ---
    metrics_group = parser.add_argument_group("Metrics Used Settings")
    metrics_group.add_argument("--no_sinkhorn_knopp", action="store_false", dest="sinkhorn_knopp", help="Disable Sinkhorn-Knopp metric")
    metrics_group.add_argument("--tangent_sinkhorn_knopp", action="store_false", help="...")
    metrics_group.add_argument("--no_mmd", action="store_false", dest="mmd", help="Disable MMD metric")
    metrics_group.add_argument("--no_epsilon_coverage", action="store_false", dest="epsilon_coverage", help="Disable Epsilon Coverage metric")
    metrics_group.add_argument("--no_epsilon_precision", action="store_false", dest="epsilon_precision", help="Disable Epsilon Precision metric")
    metrics_group.add_argument("--no_frechet_variance", action="store_false", dest="frechet_variance", help="Disable Frechet Variance metric")
    metrics_group.add_argument("--no_dispersion", action="store_false", dest="dispersion", help="Disable Dispersion metric")
    metrics_group.add_argument("--no_radial", action="store_false", dest="radial", help="Disable Radial metric")
    metrics_group.add_argument("--no_stability", action="store_false", dest="stability", help="Disable Stability metric")
    metrics_group.add_argument("--no_rfm", action="store_false", dest="rfm", help="Disable RFM metric")
    metrics_group.add_argument("--volume_scaling", action="store_false", help="...")
    metrics_group.add_argument("--cross_curvature", action="store_true", dest="cross_curvature", help="Enable Cross-Curvature metric")

    # --- Metrics Parameters ---
    metrics_param_group = parser.add_argument_group("Metrics Parameters")
    metrics_param_group.add_argument("--sinkhorn_blur", type=float, default=0.05, help="Sinkhorn blur parameter")
    metrics_param_group.add_argument("--coverage_eps_multiplier", type=float, default=1.0, help="Coverage epsilon multiplier")
    metrics_param_group.add_argument("--normalize_tangent_sinkhorn", action="store_false", help="Whether to normalize the Sinkhorn metric for curvature.")
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
    eval_group.add_argument("--early_stopping_patience", type=int, default=1000, help="Early stopping step parameter.")
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

    # --- Determine Mode & Extract Initial Means ---
    is_cartesian = args.cartesian_means is not None

    if is_cartesian:
        if args.radii is not None or args.angles is not None:
            raise ValueError("You cannot provide --radii or --angles when using --cartesian_means.")

        raw_means = json.loads(args.cartesian_means)
        num_gaussians = len(raw_means)

        if not isinstance(raw_means, list) or (
            num_gaussians > 0 and not isinstance(raw_means[0], list)
        ):
            msg = "--cartesian_means must be a JSON string of nested lists. E.g., '[[1.0, 0.0], [-1.0, 0.0]]'"
            raise ValueError(msg)

    else:
        if args.radii is None or args.angles is None:
            msg = "You must provide EITHER --cartesian_means OR both --radii and --angles."
            raise ValueError(msg)

        num_gaussians = len(args.radii)
        if len(args.angles) != num_gaussians:
            msg = "The number of arguments provided to --radii and --angles must be identical."
            raise ValueError(msg)

        # Calculate Cartesian from Polar
        raw_means = []
        for r, angle_deg in zip(args.radii, args.angles):
            angle_rad = math.radians(angle_deg)
            x = r * math.cos(angle_rad)
            y = r * math.sin(angle_rad)
            raw_means.append([x, y])

    # Validate stds and weights count
    if not (len(args.stds) == num_gaussians and len(args.weights) == num_gaussians):
        msg = f"Length mismatch: Expected {num_gaussians} stds and weights to match the defined means."
        raise ValueError(msg)

    # --- Process and Pad Means, Process Stds/Weights ---
    for g_idx in range(num_gaussians):
        identifier = f"G{g_idx}"

        mean_vector = raw_means[g_idx]
        std_str = args.stds[g_idx]
        weight = args.weights[g_idx]

        # Pad or Truncate Mean Vector based on dimensionality
        if len(mean_vector) < args.dim:
            mean_vector.extend([0.0] * (args.dim - len(mean_vector)))

        elif len(mean_vector) > args.dim:
            msg = f"Gaussian G{g_idx} mean {mean_vector} has more dimensions than the specified manifold dim ({args.dim})."
            raise ValueError(msg)

        # Round the mean vector for clean YAML output
        final_mean = [round(m, 4) for m in mean_vector]

        base_std = parse_std_string(std_str, args.dim)

        # Apply Overrides
        std = base_std
        if identifier in overrides:
            if "weight" in overrides[identifier]:
                weight = overrides[identifier]["weight"]

            if "std" in overrides[identifier]:
                std = overrides[identifier]["std"]

        all_means.append(final_mean)
        all_stds.append(std)
        all_weights.append(weight)

    # Normalize weights to sum to 1.0
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
            "origin": args.origin,
            "normalize_tangent_distributions": args.normalize_tangent_distributions,
        },
        "metrics_used": {
            "sinkhorn_knopp": args.sinkhorn_knopp,
            "tangent_sinkhorn_knopp": args.tangent_sinkhorn_knopp,
            "mmd": args.mmd,
            "epsilon_coverage": args.epsilon_coverage,
            "epsilon_precision": args.epsilon_precision,
            "frechet_variance": args.frechet_variance,
            "dispersion": args.dispersion,
            "radial": args.radial,
            "stability": args.stability,
            "rfm": args.rfm,
            "cross_curvature": args.cross_curvature,
            "volume_scaling": args.volume_scaling,
        },
        "metrics_param": {
            "sinkhorn_blur": args.sinkhorn_blur,
            "coverage_eps_multiplier": args.coverage_eps_multiplier,
            "normalize_tangent_sinkhorn": args.normalize_tangent_sinkhorn,
        },
        "save_densities": args.save_densities,
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
        "early_stopping_patience": args.early_stopping_patience,
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
    filepath = os.path.join("./riemannian-fm/configs/experiment/", args.filename)

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

    # # Print job file spec
    # with open("./riemannian-fm/configs/experiment/general_fm.yaml", "r") as fo:
    #     general_fm = yaml.safe_load(fo)

    # print("srun python train.py experiment=general_fm seed=34 \\")
    # print("\thydra.run.dir=outputs/runs/baseline/NAME \\")
    # for k, v in general_fm.items():
    #     if not isinstance(v, dict):
    #         if config[k] != v:
    #             print(f"\t{k}={v} \\")
    #         continue

    #     # print(k, v)
    #     for k2, v2 in v.items():
    #         if config[k][k2] != v2:
    #             print(f"\t{k}.{k2}={v2} \\")


if __name__ == "__main__":
    main()
