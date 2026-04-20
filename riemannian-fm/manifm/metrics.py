# import torch
# import os
# import json
# import numpy as np
# import geoopt
# from geomloss import SamplesLoss  # Requirement: pip install geomloss
# from geomstats.learning.frechet_mean import FrechetMean #Requirement: pip install geomstats
# from manifm.manifolds import Euclidean, SphereCurvature, PoincareBall
# class ManifoldMetricHandler:
#     def __init__(self, cfg):
#         """
#         Initializes the metric handler.
        
#         Args:
#             cfg: Hydra/Omegaconf object containing 'metrics_to_use' and 'save_densities'
#             manifold: The manifold object (e.g., Sphere, PoincareBall) providing .dist()
#         """
#         self.cfg = cfg
#         self.active_metrics = cfg.get("metrics_to_use", ["mse", "wasserstein", "diversity", "rfm"])
#         m_type = cfg.general.get("manifold", "euclidean").lower()
#         curvature = cfg.general.get("curvature", 1.0)
#         m_type = cfg.get("manifold", "euclidean").lower()
#         if m_type == "euclidean":
#             self.manifold = Euclidean()
#         elif m_type == "sphere":
#             self.manifold = SphereCurvature(c=curvature)
#         elif m_type == "poincare":
#             self.manifold = PoincareBall(c=curvature)
#         else:
#             raise ValueError(f"Unsupported manifold: {m_type}")

#     def calculate_wasserstein(self, x_gen, x_real, p=1, blur=0.01):
#         """
#         Calculates W_p(mu, nu) = inf E[d(x,y)^p]^(1/p).
#         Measures total distributional accuracy.
#         """
#         def geodesic_cost(x, y):
#             # Respects the Polish space (M, d) by using manifold-specific distance
#             return self.manifold.dist(x, y)**p
            

#         solver = SamplesLoss(loss="sinkhorn", p=p, cost=geodesic_cost, blur=blur)
        
#         # Result is (W_p)^p, so we return the p-th root
#         raw_dist = solver(x_gen, x_real)
#         return raw_dist ** (1/p)
    
#     def calculate_rfm_error(self, v_pred, v_target, x_t):
#         """Robustly calculates error across different manifold implementations."""
#         diff = v_pred - v_target
        
#         # 1. Determine which norm to use
#         if hasattr(self.manifold, "inner"):
#             # Works for your SphereCurvature, PoincareBall, and Euclidean
#             norm_fn = lambda u, x: torch.sqrt(self.manifold.inner(x, u, u).clamp(min=1e-12))
#         elif hasattr(self.manifold, "metric"):
#             norm_fn = lambda u, x: self.manifold.metric.norm(u, base_point=x)
#         else:
#             norm_fn = lambda u, x: torch.linalg.norm(u, dim=-1)

#         # 2. Calculate Scalar Error and Alignment
#         error = norm_fn(diff, x_t).mean()
        
#         n_p = norm_fn(v_pred, x_t).unsqueeze(-1)
#         n_t = norm_fn(v_target, x_t).unsqueeze(-1)
#         v_pred_n = v_pred / (n_p + 1e-8)
#         v_target_n = v_target / (n_t + 1e-8)

#         if hasattr(self.manifold, "inner"):
#             alignment = self.manifold.inner(x_t, v_pred_n, v_target_n).mean()
#         else:
#             alignment = (v_pred_n * v_target_n).sum(dim=-1).mean()
            
#         return error, alignment

#     def calculate_diversity(self, samples):
#         """Calculates Frechet Variance for diversity assessment."""
        
#         mean_estimator = FrechetMean(metric=self.manifold.metric) 
#         mean_estimator.fit(samples.detach().cpu().numpy())
#         variance = mean_estimator.variance(samples.detach().cpu().numpy())
#         return torch.tensor(variance)
    
#     def calculate_rfm_error(self, v_pred, v_target):
#         """
#         Compares the learned vector field with the optimal (geodesic) vector field.
#         """
#         diff = v_pred - v_target
#         error = self.manifold.metric.norm(diff).mean()
        
#         norm_pred = self.manifold.metric.norm(v_pred, keepdim=True)
#         norm_target = self.manifold.metric.norm(v_target, keepdim=True)

#         v_pred_n = v_pred / (norm_pred + 1e-8)
#         v_target_n = v_target / (norm_target + 1e-8)

#         alignment = (v_pred_n * v_target_n).sum(dim=-1).mean()
#         return error, alignment

#     def calculate_all(self, pred, target, mode="sample", step=0):
#         """
#         Main entry point for calculating and logging metrics.
        
#         mode: 'sample' for final points, 'vector' for vector fields
#         """
#         results = {}

#         if mode == "vector":
#             error, align = self.calculate_rfm_error(pred, target)
#             results["val_vec/rfm_loss"] = error
#             results["val_vec/alignment"] = align
#             results["val_vec/mse"] = torch.mean((pred - target)**2)

#         elif mode == "sample":
#             # Accuracy Metric
#             if "wasserstein" in self.active_metrics:
#                 results["val_sample/wasserstein_acc"] = self.calculate_wasserstein(pred, target)
            
#             # Diversity Metric
#             if "diversity" in self.active_metrics:
#                 results["val_sample/frechet_diversity"] = self.calculate_diversity(pred)
            
#             # Distance Metric
#             results["val_sample/geodesic_dist"] = self.manifold.dist(pred, target).mean()

#             # Save densities for later config comparison
#             if self.cfg.get("save_densities", False):
#                 self.save_density_state(pred, target, step)

#         return results

#     def save_density_state(self, x_gen, x_real, step):
#         """Saves final generated and target densities for offline analysis."""
#         path = "results/densities"
#         os.makedirs(path, exist_ok=True)
        
#         manifold_name = self.manifold.__class__.__name__
#         filename = f"{path}/{manifold_name}_step_{step:06d}.pt"
        
#         data = {
#             "x_gen": x_gen.detach().cpu(),
#             "x_real": x_real.detach().cpu(),
#             "step": step,
#             "manifold": manifold_name
#         }
#         torch.save(data, filename)

import torch
import os
from geomloss import SamplesLoss

from manifm.manifolds import Euclidean, SphereCurvature, PoincareBall


class ManifoldMetricHandler:
    def __init__(self, cfg):
        self.cfg = cfg

        self.active_metrics = cfg.get(
            "metrics_to_use",
            ["mse", "wasserstein", "diversity", "rfm"]
        )

        # --- Manifold selection ---
        self.m_type = cfg.get("manifold", "euclidean").lower()
        curvature = cfg.get("curvature", 1.0)

        if self.m_type == "euclidean":
            self.manifold = Euclidean()
        elif self.m_type == "sphere":
            self.manifold = SphereCurvature(c=curvature)
        elif self.m_type == "poincare":
            self.manifold = PoincareBall(c=curvature)
        else:
            raise ValueError(f"Unsupported manifold: {self.m_type}")

   
    def calculate_wasserstein(self, x_gen, x_real, p=1, blur=0.05):

        # --- Euclidean special case (fast + correct) ---
        if self.m_type == "euclidean":
            cost = torch.cdist(x_gen, x_real, p=2) ** p

            solver = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
            return solver(x_gen, x_real) ** (1 / p)

        # --- General manifold case ---
        def geodesic_cost(x, y):
            # Explicit pairwise expansion
            x_exp = x.unsqueeze(1)   # (N, 1, D)
            y_exp = y.unsqueeze(0)   # (1, M, D)

            return self.manifold.dist(x_exp, y_exp) ** p

        solver = SamplesLoss(loss="sinkhorn", p=p, cost=geodesic_cost, blur=blur)
        return solver(x_gen, x_real) ** (1 / p)

    
    def _norm(self, v, x):
        if self.m_type != "euclidean":
            norm = torch.sqrt(
                self.manifold.inner(x, v, v).clamp(min=1e-12)
            )
        else:
            norm = torch.linalg.norm(v, dim=-1)

        return norm 
 
   
    def calculate_rfm_error(self, v_pred, v_target, x_t):
        """
        v_pred, v_target ∈ T_{x_t}M
        """
        diff = v_pred - v_target

        error = self._norm(diff, x_t).mean()

        n_pred = self._norm(v_pred, x_t).unsqueeze(-1)
        n_target = self._norm(v_target, x_t).unsqueeze(-1)

        v_pred_n = v_pred / (n_pred + 1e-8)
        v_target_n = v_target / (n_target + 1e-8)

        if hasattr(self.manifold, "inner"):
            alignment = self.manifold.inner(x_t, v_pred_n, v_target_n).mean()
        else:
            alignment = (v_pred_n * v_target_n).sum(dim=-1).mean()

        return error, alignment


    def frechet_mean(self, x, max_iter=50, lr=0.1):
        """
        Computes intrinsic mean using Riemannian gradient descent.
        """
        # 1. Faster Euclidean Fallback
        if self.manifold.__class__.__name__ == "Euclidean":
            return x.mean(dim=0, keepdim=True)

        # 2. Robust Manifold Iteration
        mu = x[0:1].clone() 

        for _ in range(max_iter):
            v = self.manifold.logmap(mu.expand_as(x), x) 
            grad = -v.mean(dim=0, keepdim=True)
            mu = self.manifold.expmap(mu, -lr * grad)

        return mu

    
    def calculate_diversity(self, samples):
        """
        Intrinsic variance using Fréchet mean.
        """
        mu = self.frechet_mean(samples)
        
        # --- THE ONLY CRITICAL FIX ---
        # Explicitly broadcast mu to match samples shape for the manifold dist call
        mu_expanded = mu.expand_as(samples)
        variance = self.manifold.dist(samples, mu_expanded).pow(2).mean()
        return variance

    
    def calculate_all(self, pred, target, mode="sample", step=0, x_t=None):
        results = {}

        if mode == "vector":
            assert x_t is not None, "x_t required for RFM metric"

            error, align = self.calculate_rfm_error(pred, target, x_t)

            results["val_vec/rfm_loss"] = error
            results["val_vec/alignment"] = align
            results["val_vec/mse"] = torch.mean((pred - target) ** 2)

        elif mode == "sample":

            if "wasserstein" in self.active_metrics:
                results["val_sample/wasserstein"] = self.calculate_wasserstein(pred, target)

            if "diversity" in self.active_metrics:
                results["val_sample/diversity"] = self.calculate_diversity(pred)

            if pred.shape == target.shape:
                if self.m_type == "euclidean":
                    dist_val = torch.linalg.norm(pred - target, dim=-1).mean()
                else:
                    dist_val = self.manifold.dist(pred, target).mean()
                
                results["val_sample/geodesic_dist"] = dist_val
            if self.cfg.get("save_densities", False):
                self.save_density_state(pred, target, step)

        return results

   
    def save_density_state(self, x_gen, x_real, step):
        path = "results/densities"
        os.makedirs(path, exist_ok=True)

        manifold_name = self.manifold.__class__.__name__

        data = {
            "x_gen": x_gen.detach().cpu(),
            "x_real": x_real.detach().cpu(),
            "step": step,
            "manifold": manifold_name,
            "curvature": getattr(self.manifold, "c", None),
            "num_samples": x_gen.shape[0],
        }

        torch.save(data, f"{path}/{manifold_name}_step_{step:06d}.pt")
