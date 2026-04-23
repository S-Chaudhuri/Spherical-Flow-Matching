import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from manifm.metrics import ManifoldMetricHandler
from manifm.manifolds import PoincareBall, SphereCurvature, Euclidean
from manifm.datasets import HyperbolicDatasetPair
from torch.utils.data import DataLoader

def get_test_config(manifold_type):
    return {
        "manifold": manifold_type,
        "curvature": 1.0,
        "metrics_to_use": ["wasserstein", "diversity", "rfm"],
        "save_densities": False,
        "monitor_metric": "wasserstein"
    }


def sample_euclidean(n=500, dim=2):
    x = torch.randn(n, dim)
    y = torch.randn(n, dim) + 1.5
    return x, y

def sample_sphere(n=500, dim=3):
    m = SphereCurvature(c=1.0)
    # Start with random normal
    x = torch.randn(n, dim)
    y = torch.randn(n, dim) + 0.5
    # Project onto surface
    return m.projx(x), m.projx(y)

def sample_poincare(n=1000, dim=2):
    from manifm.manifolds import PoincareBall
    manifold = PoincareBall()

    x = torch.stack([
        manifold.wrapped_normal(dim, mean=torch.zeros(dim), std=0.3)
        for _ in range(n)
    ])

    y = torch.stack([
        manifold.wrapped_normal(dim, mean=torch.ones(dim)*0.5, std=0.3)
        for _ in range(n)
    ])

    return x, y

def generate_vector_field(x):
    # Simple flow toward a target
    v_target = -0.5 * x 
    v_pred = v_target + 0.02 * torch.randn_like(x)
    return v_pred, v_target


def plot_2d(x, y, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y[:, 0], y[:, 1], alpha=0.4, label="Real (Target)", c='blue')
    plt.scatter(x[:, 0], x[:, 1], alpha=0.4, label="Generated", c='red')
    plt.legend()
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path)
    plt.close()

def plot_metrics(logs, title, save_path):
    plt.figure(figsize=(8, 5))
    for key, values in logs.items():
        if values:
            plt.plot(values, label=key, marker='o')
    plt.legend()
    plt.title(f"Metric Trends: {title}")
    plt.xlabel("Step")
    plt.savefig(save_path)
    plt.close()


def run_test(manifold_type):
    print(f"\n=== Testing {manifold_type} ===")
    cfg = get_test_config(manifold_type)
    metrics = ManifoldMetricHandler(cfg)

    # Use keys matching your Handler's internal logic
    logs = {"wasserstein": [], "diversity": [], "alignment": [], "rfm_loss": []}
    

    for step in range(5):
        if manifold_type == "euclidean": 
            x_gen, x_real = sample_euclidean()
        elif manifold_type == "sphere": 
            x_gen, x_real = sample_sphere()
        elif manifold_type == "poincare": 
            hyperbolic_ds = HyperbolicDatasetPair()
            loader = DataLoader(hyperbolic_ds, batch_size=128)
            
            batch = next(iter(loader))
            
            x_gen, x_real = batch["x1"], batch["x0"] 

    # for step in range(5):
    #     if manifold_type == "euclidean": x_gen, x_real = sample_euclidean()
    #     elif manifold_type == "sphere": x_gen, x_real = sample_sphere()
    #     elif manifold_type == "poincare": x_gen, x_real = HyperbolicDatasetPair()#sample_poincare()

        # 1. Test Sample Mode (Accuracy/Diversity)
        sample_results = metrics.calculate_all(x_gen, x_real, mode="sample", step=step)
        
        # 2. Test Vector Mode (Flow Alignment)
        v_pred, v_target = generate_vector_field(x_gen)
        vec_results = metrics.calculate_all(v_pred, v_target, mode="vector", x_t=x_gen)

        # Extraction (Matches your Handler's log strings)
        logs["wasserstein"].append(sample_results.get("val_sample/wasserstein", torch.tensor(0.0)).item())
        logs["diversity"].append(sample_results.get("val_sample/diversity", torch.tensor(0.0)).item())
        logs["alignment"].append(vec_results.get("val_vec/alignment", torch.tensor(0.0)).item())
        logs["rfm_loss"].append(vec_results.get("val_vec/rfm_loss", torch.tensor(0.0)).item())

        print(f"Step {step} | W1: {logs['wasserstein'][-1]:.4f} | Align: {logs['alignment'][-1]:.4f}")

    # Save results
    os.makedirs("results/plots", exist_ok=True)
    
    # 2D Plotting (For Euclidean and Poincare)
    if x_gen.shape[1] == 2:
        plot_2d(
            x_gen.detach().cpu().numpy(),
            x_real.detach().cpu().numpy(),
            f"{manifold_type.capitalize()} Distribution Match",
            f"results/plots/{manifold_type}_samples.png",
        )

    plot_metrics(logs, manifold_type.capitalize(), f"results/plots/{manifold_type}_metrics.png")

if __name__ == "__main__":
    for manifold in ["euclidean", "sphere", "poincare"]:
        try:
            run_test(manifold)
        except Exception as e:
            print(f" Error testing {manifold}: {e}")

    print("\n Testing complete. Check results/plots/ for PNG files.")
