import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def extract_data(data):
    """Safely extract and convert PyTorch tensors to numpy arrays."""
    return {
        "x0": data["eval_x0"].numpy(),
        "x1": data["eval_x1"].numpy(),
        "x1_hat": data["x1_hat"].numpy(),
        "x_t": data["x_t"].numpy(),  # Shape: (T, N, D)
        "u_t": data["u_t"].numpy(),
        "vtheta": data["vtheta"].numpy(),
        "eval_t": data["eval_t"].numpy(),
    }


def format_title(meta, subtitle):
    """Formats the title to include metadata for easy verification."""
    meta_str = " | ".join(
        [
            f"{k}: {v}"
            for k, v in meta.items()
            if k != "cfg_yaml" and not isinstance(v, (list, tuple, dict))
        ]
    )
    return f"\n\n\n{subtitle}\n{meta_str}"


def plot_euclidean(data_dict, meta, save_path=None):
    """Standard 2D Euclidean Plotting"""
    d = data_dict
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(format_title(meta, "Euclidean Flow Matching"), fontsize=12, y=1.05)

    # Panel 1: Distributions
    axes[0].scatter(d["x0"][:, 0], d["x0"][:, 1], c="gray", alpha=0.3, label="Source (x0)", s=10)
    axes[0].scatter(
        d["x1"][:, 0], d["x1"][:, 1], c="blue", alpha=0.5, label="True Target (x1)", s=15
    )
    axes[0].scatter(
        d["x1_hat"][:, 0], d["x1_hat"][:, 1], c="red", alpha=0.5, label="Generated (x1_hat)", s=15
    )
    axes[0].set_title("Distribution Matching")
    axes[0].legend()
    axes[0].axis("equal")

    # Panel 2: Trajectories
    axes[1].scatter(d["x0"][:, 0], d["x0"][:, 1], c="gray", s=10)
    axes[1].scatter(d["x1"][:, 0], d["x1"][:, 1], c="blue", s=10)
    for n in range(min(50, d["x_t"].shape[1])):
        axes[1].plot(d["x_t"][:, n, 0], d["x_t"][:, n, 1], c="black", alpha=0.3, linewidth=1)
    axes[1].set_title("True Geodesic Trajectories")
    axes[1].axis("equal")

    # Panel 3: Vector Field
    mid_idx = len(d["eval_t"]) // 2
    axes[2].quiver(
        d["x_t"][mid_idx, :, 0],
        d["x_t"][mid_idx, :, 1],
        d["u_t"][mid_idx, :, 0],
        d["u_t"][mid_idx, :, 1],
        color="blue",
        alpha=0.5,
        label="True Field (u_t)",
    )
    axes[2].quiver(
        d["x_t"][mid_idx, :, 0],
        d["x_t"][mid_idx, :, 1],
        d["vtheta"][mid_idx, :, 0],
        d["vtheta"][mid_idx, :, 1],
        color="red",
        alpha=0.5,
        label="Predicted (vtheta)",
    )
    axes[2].set_title(f"Field Alignment at t={d['eval_t'][mid_idx]:.2f}")
    axes[2].legend()
    axes[2].axis("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_poincare(data_dict, meta, save_path=None):
    """Replicates the codebase's plot_poincare style with the bounding circle."""
    d = data_dict
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(format_title(meta, "Poincaré Disk Flow Matching"), fontsize=12, y=1.05)

    for ax in axes:
        ax.add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect("equal")
        ax.axis("off")

    # Panel 1: Distributions
    axes[0].scatter(d["x0"][:, 0], d["x0"][:, 1], c="gray", alpha=0.3, label="Source", s=10)
    axes[0].scatter(d["x1"][:, 0], d["x1"][:, 1], c="blue", alpha=0.5, label="True Target", s=15)
    axes[0].scatter(
        d["x1_hat"][:, 0], d["x1_hat"][:, 1], c="red", alpha=0.5, label="Generated", s=15
    )
    axes[0].set_title("Distribution Matching")
    axes[0].legend(loc="upper right")

    # Panel 2: Trajectories
    axes[1].scatter(d["x0"][:, 0], d["x0"][:, 1], c="gray", s=10)
    axes[1].scatter(d["x1"][:, 0], d["x1"][:, 1], c="blue", s=10)
    for n in range(min(100, d["x_t"].shape[1])):
        axes[1].plot(d["x_t"][:, n, 0], d["x_t"][:, n, 1], color="grey", linewidth=0.5)
    axes[1].set_title("Geodesic Trajectories")

    # Panel 3: Vector Field
    mid_idx = len(d["eval_t"]) // 2
    axes[2].quiver(
        d["x_t"][mid_idx, :, 0],
        d["x_t"][mid_idx, :, 1],
        d["u_t"][mid_idx, :, 0],
        d["u_t"][mid_idx, :, 1],
        color="blue",
        alpha=0.5,
        label="u_t",
    )
    axes[2].quiver(
        d["x_t"][mid_idx, :, 0],
        d["x_t"][mid_idx, :, 1],
        d["vtheta"][mid_idx, :, 0],
        d["vtheta"][mid_idx, :, 1],
        color="red",
        alpha=0.5,
        label="vtheta",
    )
    axes[2].set_title(f"Field Alignment at t={d['eval_t'][mid_idx]:.2f}")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_sphere_2d(data_dict, meta, save_path=None):
    """Plots a 1D circle embedded in 2D space."""
    d = data_dict
    curvature = meta.get("curvature", 1.0)
    R = 1.0 / np.sqrt(curvature) if curvature > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        format_title(meta, f"1D Spherical Flow Matching (S1 in R2, R={R:.2f})"), fontsize=12, y=1.05
    )

    for ax in axes:
        # Draw the 2D circular boundary
        ax.add_patch(
            plt.Circle((0, 0), R, color="lightblue", fill=False, linewidth=2, linestyle="--")
        )
        ax.set_xlim([-R * 1.2, R * 1.2])
        ax.set_ylim([-R * 1.2, R * 1.2])
        ax.set_aspect("equal")
        ax.axis("off")

    # Panel 1: Distributions
    axes[0].scatter(d["x0"][:, 0], d["x0"][:, 1], c="gray", alpha=0.5, label="Source", s=10)
    axes[0].scatter(d["x1"][:, 0], d["x1"][:, 1], c="blue", alpha=0.5, label="True Target", s=15)
    axes[0].scatter(
        d["x1_hat"][:, 0], d["x1_hat"][:, 1], c="red", alpha=0.5, label="Generated", s=15
    )
    axes[0].set_title("Distribution Matching")
    axes[0].legend(loc="upper right")

    # Panel 2: Trajectories
    axes[1].scatter(d["x0"][:, 0], d["x0"][:, 1], c="gray", s=10, zorder=3)
    axes[1].scatter(d["x1"][:, 0], d["x1"][:, 1], c="blue", s=10, zorder=3)
    for n in range(min(50, d["x_t"].shape[1])):
        axes[1].plot(
            d["x_t"][:, n, 0], d["x_t"][:, n, 1], color="grey", alpha=0.5, linewidth=1, zorder=1
        )
    axes[1].set_title("Geodesic Trajectories")

    # Panel 3: Vector Field
    mid_idx = len(d["eval_t"]) // 2
    idx = np.random.choice(d["x_t"].shape[1], min(30, d["x_t"].shape[1]), replace=False)
    xt_m, ut_m, vt_m = d["x_t"][mid_idx][idx], d["u_t"][mid_idx][idx], d["vtheta"][mid_idx][idx]

    axes[2].quiver(
        xt_m[:, 0],
        xt_m[:, 1],
        ut_m[:, 0],
        ut_m[:, 1],
        color="blue",
        alpha=0.5,
        label="u_t",
        zorder=2,
    )
    axes[2].quiver(
        xt_m[:, 0],
        xt_m[:, 1],
        vt_m[:, 0],
        vt_m[:, 1],
        color="red",
        alpha=0.5,
        label="vtheta",
        zorder=3,
    )
    axes[2].set_title(f"Field Alignment (t={d['eval_t'][mid_idx]:.2f})")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_sphere_3d(data_dict, meta, save_path=None):
    """Plots a 2D surface embedded in 3D space with a wireframe."""
    d = data_dict
    curvature = meta.get("curvature", 1.0)
    R = 1.0 / np.sqrt(curvature) if curvature > 0 else 1.0

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        format_title(meta, f"2D Spherical Flow Matching (S2 in R3, R={R:.2f})"), fontsize=12, y=1.05
    )

    axes = [
        fig.add_subplot(131, projection="3d"),
        fig.add_subplot(132, projection="3d"),
        fig.add_subplot(133, projection="3d"),
    ]

    # Draw the wireframe sphere on all panels
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    u, v = np.meshgrid(u, v)
    xs = R * np.cos(u) * np.sin(v)
    ys = R * np.sin(u) * np.sin(v)
    zs = R * np.cos(v)

    for ax in axes:
        ax.plot_wireframe(xs, ys, zs, color="lightblue", alpha=0.1, linewidth=0.5)
        ax.set_xlim([-R, R])
        ax.set_ylim([-R, R])
        ax.set_zlim([-R, R])
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

    # Panel 1: Distributions
    axes[0].scatter(
        d["x0"][:, 0], d["x0"][:, 1], d["x0"][:, 2], c="gray", alpha=0.3, label="Source"
    )
    axes[0].scatter(
        d["x1"][:, 0], d["x1"][:, 1], d["x1"][:, 2], c="blue", alpha=0.5, label="True Target"
    )
    axes[0].scatter(
        d["x1_hat"][:, 0],
        d["x1_hat"][:, 1],
        d["x1_hat"][:, 2],
        c="red",
        alpha=0.5,
        label="Generated",
    )
    axes[0].set_title("Distribution Matching")
    axes[0].legend()

    # Panel 2: Trajectories
    for n in range(min(50, d["x_t"].shape[1])):
        axes[1].plot(
            d["x_t"][:, n, 0],
            d["x_t"][:, n, 1],
            d["x_t"][:, n, 2],
            color="grey",
            alpha=0.5,
            linewidth=1,
        )
    axes[1].set_title("Geodesic Trajectories")

    # Panel 3: Vector Field
    mid_idx = len(d["eval_t"]) // 2
    idx = np.random.choice(d["x_t"].shape[1], min(30, d["x_t"].shape[1]), replace=False)
    xt_m, ut_m, vt_m = d["x_t"][mid_idx][idx], d["u_t"][mid_idx][idx], d["vtheta"][mid_idx][idx]

    axes[2].quiver(
        xt_m[:, 0],
        xt_m[:, 1],
        xt_m[:, 2],
        ut_m[:, 0],
        ut_m[:, 1],
        ut_m[:, 2],
        color="blue",
        alpha=0.5,
        label="u_t",
        length=R * 0.2,
        normalize=True,
    )
    axes[2].quiver(
        xt_m[:, 0],
        xt_m[:, 1],
        xt_m[:, 2],
        vt_m[:, 0],
        vt_m[:, 1],
        vt_m[:, 2],
        color="red",
        alpha=0.5,
        label="vtheta",
        length=R * 0.2,
        normalize=True,
    )
    axes[2].set_title(f"Field Alignment (t={d['eval_t'][mid_idx]:.2f})")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Visualize Geometric Flow Matching Results")
    parser.add_argument("--file", "-f", required=True, help="Path to the .pt file")
    parser.add_argument("--save", "-s", action="store_true", help="Save the plot to a file")
    parser.add_argument("--out", "-o", default=".results/images/run_analysis.png", help="Output filename if saving")

    args = parser.parse_args()
    # fmt: on

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    print(f"Loading data from {args.file}...")
    raw_data = torch.load(args.file, map_location="cpu", weights_only=True)
    meta = raw_data.get("meta", {})
    data = extract_data(raw_data)

    # Route to the correct geometric plotting function
    manifold_type = meta.get("manifold_type", "").lower()
    dim = data["x0"].shape[1]

    save_path = args.out if args.save else None

    if "poincare" in manifold_type:
        print(f"Detected Poincaré manifold (Dim={dim}). Plotting in the hyperbolic disk...")
        plot_poincare(data, meta, save_path)
    elif "sphere" in manifold_type:
        if dim == 2:
            print(f"Detected Spherical manifold (Dim={dim}). Plotting 2D circle...")
            plot_sphere_2d(data, meta, save_path)
        else:
            print(f"Detected Spherical manifold (Dim={dim}). Plotting 3D wireframe sphere...")
            plot_sphere_3d(data, meta, save_path)
    else:
        print(f"Defaulting to Euclidean flat space plotting (Dim={dim})...")
        plot_euclidean(data, meta, save_path)


if __name__ == "__main__":
    main()
