from __future__ import annotations

import os
import argparse
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf



BASE_DIR = Path(__file__).resolve().parent

def get_metrics(run_dir, run_glob, out_dir, default_metrics=None):
    run_dirs = collect_run_dirs(run_dir, run_glob)

    if not run_dirs:
        raise ValueError('No runs specified. Fill RUN_DIRS and/or RUN_GLOBS in the cell above.')

    out_dir = Path(out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (BASE_DIR / out_dir).resolve()

    metrics_list = {}
    
    
    for i, run_dir in enumerate(run_dirs, start=1):
        # print('\n' + '=' * 100)
        # print(f'[{i}/{len(run_dirs)}] Run: {run_dir}')
        
        # metadata from run (manifold, curvature, dimension)
        meta = load_general_from_hydra(run_dir)

        paths = resolve_run_paths(run_dir)

        missing = []
        if not paths.run_dir.exists():
            missing.append(f'missing run dir: {paths.run_dir}')
        if not paths.artifacts_pt.exists():
            missing.append(f'missing artifacts file: {paths.artifacts_pt}')
        if not paths.metrics_json.exists():
            missing.append(f'missing metrics file: {paths.metrics_json}')

        if missing:
            print('SKIP (missing files):')
            for m in missing:
                print(f'  - {m}')
            continue

        # --- Metrics ---
        # print('\nMetrics:')
        metrics = load_metrics(paths.metrics_json, meta)
        # print_metrics(metrics, key_subset=default_metrics)

        # --- Plot ---
        label = safe_run_label(paths.run_dir)
        out_file = out_dir / f'{label}_analysis.pdf'
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # print(f'\nGenerating plot -> {out_file}')
        visualize_pt_file(paths.artifacts_pt, run_dir=paths.run_dir, meta=meta, save_path=out_file)
        metrics_list[label] = metrics
    return metrics_list

def find_visualize_script(start_dir: str | Path) -> Path:
    """Locate visualize_run.py starting from a directory, searching up the tree.

    This is mainly useful in notebooks/environments where the working directory
    may differ from the directory containing this file.
    """

    start_dir_path = Path(start_dir).expanduser().resolve()
    for base in [start_dir_path, *start_dir_path.parents]:
        direct = base / "visualize_run.py"
        in_results = base / "results" / "visualize_run.py"
        if direct.exists():
            return direct
        if in_results.exists():
            return in_results

    raise FileNotFoundError(
        "Could not locate visualize_run.py from the provided start directory. "
        "Try opening/running from riemannian-fm/results/."
    )


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    artifacts_pt: Path
    metrics_json: Path


def resolve_run_paths(run_dir: str | Path, base_dir: Path | None = None) -> RunPaths:
    """Resolve a run directory into expected artifact/metric paths."""

    base_dir = base_dir or BASE_DIR
    run_dir_path = Path(run_dir).expanduser()
    if not run_dir_path.is_absolute():
        run_dir_path = base_dir / run_dir_path
    run_dir_path = run_dir_path.resolve()

    artifacts_pt = run_dir_path / "artifacts" / "final_fixed_eval_outputs.pt"
    metrics_json = run_dir_path / "metrics.json"
    return RunPaths(run_dir=run_dir_path, artifacts_pt=artifacts_pt, metrics_json=metrics_json)


def collect_run_dirs(
    run_dirs: list[str | Path],
    run_globs: list[str],
    base_dir: Path | None = None,
) -> list[Path]:
    """Collect run directories from explicit paths and/or glob patterns."""

    base_dir = base_dir or BASE_DIR
    collected: list[Path] = []

    for rd in run_dirs:
        collected.append(Path(rd).expanduser())

    for pattern in run_globs:
        # Globs are interpreted relative to base_dir unless absolute.
        if Path(pattern).is_absolute():
            matches = [Path(p) for p in glob.glob(pattern)]
        else:
            matches = list(base_dir.glob(pattern))
        collected.extend(sorted(matches))

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in collected:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    return unique


def safe_run_label(run_dir: str | Path) -> str:
    """Build a short label for filenames (e.g. baseline__poi_d3_c1)."""

    run_dir_path = Path(run_dir)
    parts = [run_dir_path.parent.name, run_dir_path.name]
    label = "__".join([p for p in parts if p])
    return label or run_dir_path.name


def load_metrics(metrics_path: str | Path, meta: dict) -> dict:
    metrics_path = Path(metrics_path).expanduser().resolve()
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    # Merge metadata into metrics
    for k, v in meta.items():
        metrics[f"{k}"] = v
    return metrics


def print_metrics(metrics: dict, key_subset: list[str] | None = None) -> None:
    if key_subset is not None and key_subset != []:
        print("Key metrics:")
        for k, v in metrics.items():
            if k in key_subset:
                print(f"  {k}: {v}")
    else:
        print("\nAll metrics (sorted):")
        for k in sorted(metrics.keys()):
            print(f"  {k}: {metrics[k]}")


def load_general_from_hydra(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    general = cfg.get("general")
    return OmegaConf.to_container(general, resolve=True) if general is not None else {}

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


def visualize_pt_file(pt_file: str | Path, run_dir: str | Path, meta: dict, save_path: str | Path | None = None) -> dict:
    """Load a saved eval .pt file and generate the appropriate plot.
    """

    pt_path = Path(pt_file).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"Eval outputs not found: {pt_path}")

    raw_data = torch.load(pt_path, map_location="cpu", weights_only=True)
    data = extract_data(raw_data)

    manifold_type = meta.get("manifold", "").lower()
    dim = data["x0"].shape[1]

    save_path_str = str(Path(save_path)) if save_path is not None else None

    if "poincare" in manifold_type:
        #print(f"Detected Poincaré manifold (Dim={dim}). Plotting in the hyperbolic disk...")
        if dim == 2:
            plot_poincare(data, meta, save_path_str)
    elif "sphere" in manifold_type:
        if dim == 2:
            #print(f"Detected Spherical manifold (Dim={dim}). Plotting 2D circle...")
            plot_sphere_2d(data, meta, save_path_str)
        elif dim == 3:
            #print(f"Detected Spherical manifold (Dim={dim}). Plotting 3D wireframe sphere...")
            plot_sphere_3d(data, meta, save_path_str)
    else:
        if dim == 2:
        #print(f"Defaulting to Euclidean flat space plotting (Dim={dim})...")
            plot_euclidean(data, meta, save_path_str)

    return meta


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Visualize Geometric Flow Matching Results")
    parser.add_argument("--file", "-f", required=True, help="Path to the .pt file")
    parser.add_argument("--run-dir", "-r", required=True, help="Path to the run directory")
    parser.add_argument("--save", "-s", action="store_true", help="Save the plot to a file")
    parser.add_argument("--out", "-o", default=".results/images/run_analysis.png", help="Output filename if saving")

    args = parser.parse_args()
    meta = load_general_from_hydra(args.run_dir)
    # fmt: on

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    save_path = args.out if args.save else None
    print(f"Loading data from {args.file}...")
    visualize_pt_file(args.file, args.run_dir, meta=meta, save_path=save_path)


if __name__ == "__main__":
    main()
