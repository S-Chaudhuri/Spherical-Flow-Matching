# riemannian-fm/manifm/metrics.py

import torch
import os
from geomloss import SamplesLoss

from manifm.manifolds import Euclidean, SphereCurvature, PoincareBall
from geoopt import ManifoldTensor


class ManifoldMetricHandler:
    """
    Class to handle all metrics systematically across curvatures
    """

    def __init__(self, cfg):
        self.cfg = cfg  # store configuration dictionary

        self.active_metrics = cfg.get(  # choose metrics
            "metrics_to_use",
            [
                "sinkhorn_knopp",
                "mmd",
                "epsilon_coverage",
                "epsilon_precision",
                "frechet_variance",
                "dispersion",
                "radial",
                "stability",
                "rfm",
            ],
        )
        self.cross_curvature = cfg.get("cross_curvature", False)  # whether to normalize metrics for better cross-curvature comparison
        self.m_type = cfg.get("manifold", "euclidean").lower()
        self.curvature = cfg.get("curvature", 1.0)
        self.origin = cfg.get(
            "origin", None
        )  #! How to define the origin in the first place?

        if self.m_type == "euclidean":
            self.manifold = Euclidean()
            self.kappa = 0.0  # flat geometry
        elif self.m_type == "sphere":
            self.manifold = SphereCurvature(c=self.curvature)
            self.kappa = self.curvature  # positive curvature (spherical)
        elif self.m_type == "poincare":
            self.manifold = PoincareBall(c=self.curvature)
            self.kappa = -self.curvature  # negative curvature (hyperbolic)
        else:
            raise ValueError(f"unsupported manifold: {self.m_type}")


    def get_origin(self, x):
        """
        Return the fixed reference origin used for radial metrics:
        if cfg["origin"] is provided, we use that; otherwise we
        use a canonical origin for the chosen manifold
        """
        if self.origin is not None:
            origin = self.origin
            if not torch.is_tensor(origin):
                origin = torch.tensor(origin, device=x.device, dtype=x.dtype)
            return origin.to(device=x.device, dtype=x.dtype).view(1, -1)

        if self.m_type == "euclidean":
            return torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)

        if self.m_type == "poincare":
            return torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)

        #! See if this lines up with the SphereCurvature class
        if self.m_type == "sphere":
            radius = 1.0 / torch.sqrt(
                torch.as_tensor(self.curvature, device=x.device, dtype=x.dtype)
            )
            origin = torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
            origin[:, 0] = radius  #should we use freichet mean here instead? or does it not matter as long as it's fixed? should be fine as long as it's fixed.
            return origin

        raise ValueError(f"origin not defined for manifold: {self.m_type}")
    

    def scaled_dist(self, x, y):
        if self.m_type == "euclidean":
            return torch.cdist(x, y, p=2)

        d = self.manifold.dist(x, y)

        if self.kappa != 0:
            scale = torch.sqrt(torch.tensor(abs(self.kappa), device=d.device, dtype=d.dtype))
            d = d * scale

        return d
    

    def radial_values(self, x):
        origin = self.get_origin(x).expand_as(x)
        if self.cross_curvature:
            return self.scaled_dist(origin, x)
        else:
            return self.manifold.dist(origin, x)


    def calculate_sinkhorn_divergence(
        self,
        x_gen,
        x_real,
        p = 1,  # p = 1 gives Earth Mover's Distance (Wasserstein exponent)
        blur = 0.05,
    ):
        """
        A measure of distributional misalignment: the
        Sinkhorn-Knopp algorithm approximates the Wasserstein
        distance efficient

        x_gen: generated samples on the manifold
        x_real: target samples on the manifold
        p: transport exponent
        blur: entropic regularization parameter for geomloss
        normalize: divide by target dispersion for cross-curvature comparison
        """
        if blur is None:
            #! Blur should also be normalized across curvatures! Take this into account
            blur = self.cfg.get("sinkhorn_blur", 0.05)
        if self.cross_curvature and self.kappa != 0: #normalizing for better cross-curvature comparison
            blur = blur * (abs(self.kappa) ** 0.5)
        if self.m_type == "euclidean":
            solver = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
            val = solver(x_gen, x_real)
        else:

            def geodesic_cost(x, y):
                # do an explicit pairwise comparison
                x_exp = x.unsqueeze(1)  # (N, 1, D)
                y_exp = y.unsqueeze(0)  # (1, M, D)
                
                if self.cross_curvature:
                   d = self.scaled_dist(x_exp, y_exp) #normalizing for better cross-curvature comparison
                   return d ** p
                else:
                   return self.manifold.dist(x_exp, y_exp) ** p

            solver = SamplesLoss(
                loss = "sinkhorn",
                p = p,
                blur = blur,
                debias = True, # debiasing so we get the Sinkhorn divergence by Feydy et al. (2019)
                cost = geodesic_cost,
            )
            val = solver(x_gen, x_real)

        val = torch.clamp(val, min=0.0) ** (1.0 / p)
        #! TODO: How to normalize Wasserstein/Sinkhorn-Knopp across curvatures?
        #! Add an if-s with boolean normalize parameter
        #! Also add it to the calculate_all() function below once chosen
        return val

    #! Add decomposed Sinkhorn Knopp for radial and angular


    def pairwise_dist(self, x, y):
        if self.m_type == "euclidean":
            return torch.cdist(x, y, p=2)

        x_exp = x.unsqueeze(1)
        y_exp = y.unsqueeze(0)
        
        if self.cross_curvature:
            return self.scaled_dist(x_exp, y_exp) #normalizing for better cross-curvature comparison
        else:
            return self.manifold.dist(x_exp, y_exp)


    def calculate_mmd(self, x_gen, x_real, sigma=None):
        """
        MMD with geodesic RBF kernel
        """
        # generated samples pairwise distances
        dxx = self.pairwise_dist(x_gen, x_gen)
        # real samples pairwise distances
        dyy = self.pairwise_dist(x_real, x_real)
        # distances between generated and real
        dxy = self.pairwise_dist(x_gen, x_real)

        if sigma is None:  # use adaptive sigma if not provided
            # flatten all distances and remove non-zero
            vals = dxy.detach().flatten()
            vals = vals[vals > 0]
            sigma = torch.median(vals)  # = median of positive distances; then clamp
            sigma = torch.clamp(sigma, min=1e-6)

            # RBF kernel for generated, real, and cross samples
        kxx = torch.exp(-(dxx**2) / (2 * sigma**2))
        kyy = torch.exp(-(dyy**2) / (2 * sigma**2))
        kxy = torch.exp(-(dxy**2) / (2 * sigma**2))

        # return MMD statistic
        return kxx.mean() + kyy.mean() - 2 * kxy.mean()


    def calculate_epsilon_coverage(
        self,
        x_gen,
        x_real,
        eps=None,
        eps_multiplier=None,
    ):
        """
        Measures how much of the target support is reached by generated samples
        """
        d = self.pairwise_dist(x_real, x_gen)

        if eps_multiplier is None:
            eps_multiplier = self.cfg.get("coverage_eps_multiplier", 1.0)

        if eps is None:
            d_real = self.pairwise_dist(x_real, x_real)
            n = x_real.shape[0]
            d_real = d_real + torch.eye(n, device=x_real.device) * 1e9
            eps = d_real.min(dim=1).values.median()
            eps = eps_multiplier * eps

        nearest = d.min(dim=1).values
        return (nearest <= eps).float().mean()


    def calculate_epsilon_precision(
        self,
        x_gen,
        x_real,
        eps=None,
        eps_multiplier=None,
    ):
        """
        Measures how many generated samples lie within epsilon of the target support
        """
        d = self.pairwise_dist(x_gen, x_real)

        if eps_multiplier is None:
            eps_multiplier = self.cfg.get("coverage_eps_multiplier", 1.0)

        if eps is None:
            d_real = self.pairwise_dist(x_real, x_real)
            n = x_real.shape[0]
            d_real = d_real + torch.eye(n, device=x_real.device) * 1e9
            eps = d_real.min(dim=1).values.median()
            eps = eps_multiplier * eps

        nearest = d.min(dim=1).values
        return (nearest <= eps).float().mean()


    # def _norm(self, v, x):
    #     if self.m_type != "euclidean":
    #         norm = torch.sqrt(self.manifold.inner(x, v, v).clamp(min = 1e-12))
    #     else:
    #         norm = torch.linalg.norm(v, dim = -1)
    #     return norm


    def _norm(self, v, x):
        """
        Computes tangent-vector norms using the Riemannian metric
        """
        #! Verify whether this is correct...
        if self.m_type == "euclidean":
            return torch.linalg.norm(v, dim=-1)

        inner = self.manifold.inner(x, v, v)
        return torch.sqrt(torch.clamp(inner, min=1e-12))

    def frechet_mean(self, x, max_iter=50, lr=0.1):
        if self.m_type == "euclidean":
            return x.mean(dim=0, keepdim=True)

        with torch.no_grad(): 
            mu = self.get_origin(x).clone()  # start at origin, changed from random sample

            for _ in range(max_iter):
                v = self.manifold.logmap(mu.expand_as(x), x)
                step = v.mean(dim=0, keepdim=True)
                mu = self.manifold.expmap(mu, lr * step) # Assuming it maps from mu to x.

                if hasattr(self.manifold, "projx"):
                    mu = self.manifold.projx(mu)
        return mu


    def calculate_frechet_variance(self, samples):
        """
        Calculate Frechet variance using Frechet mean
        """
        mu = self.frechet_mean(samples)
        mu_expanded = mu.expand_as(samples)
        return self.manifold.dist(samples, mu_expanded).pow(2).mean()
        # not using this as, we would need to normalize logmap, for statistics not needed.
        #return self.scaled_dist(samples, mu_expanded).pow(2).mean() #normalizing for better cross-curvature comparison  


    def calculate_dispersion(self, samples):
        """
        Pairwise dispersion calculation: simpler version of Frechet variance
        """
        d = self.pairwise_dist(samples, samples)
        n = samples.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=samples.device)
        return d[mask].mean()


    def calculate_dispersion_ratio(self, x_pred, x_target):
        """
        Diversity: Calculate dispersion ratio of predicted with target dispersion
        """
        gen_disp = self.calculate_dispersion(x_pred)
        real_disp = self.calculate_dispersion(x_target)
        return gen_disp / torch.clamp(real_disp, min=1e-8)


    def calculate_vector_norm_stats(self, v, x):
        """
        Stability: vector field norm statistics
        """
        norms = self._norm(v, x)

        return {
            "mean": norms.mean(),
            "std": norms.std(unbiased=False),
            "max": norms.max(),
            "p95": torch.quantile(norms, 0.95),
        }

    #! Do we need divergence or some more in-depth metrics?

    def calculate_tangency_violation(self, x, v):
        """
        Validity: absolute and relative tangent-space violation
        """
        if hasattr(self.manifold, "proju"):
            v_proj = self.manifold.proju(x, v)
            violation = self._norm(v - v_proj, x)
            norm = self._norm(v, x)
            relative = violation / torch.clamp(norm, min=1e-8)

            return {
                "absolute": violation.mean(),
                "relative": relative.mean(),
            }

        if self.m_type == "sphere":
            violation = torch.abs((x * v).sum(dim=-1))
            norm = torch.linalg.norm(v, dim=-1)
            relative = violation / torch.clamp(norm, min=1e-8)

            return {
                "absolute": violation.mean(),
                "relative": relative.mean(),
            }

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return {
            "absolute": zero,
            "relative": zero,
        }


    def finite_fraction(self, x):
        """
        Stability: Fraction of samples without NaN or inf
        """
        return torch.isfinite(x).all(dim=-1).float().mean()


    def calculate_rfm_loss(
        self,
        v_pred,
        v_target,
        x_t,
    ):
        """
        Computes RFM vector-field error

        v_pred: predicted tangent vector at x_t
        v_target: target tangent vector at x_t
        x_t: manifold points where the field is evaluated
        """
        diff = v_pred - v_target

        if self.m_type == "euclidean":
            loss = (diff**2).sum(dim=-1).mean()
        else:
            loss = self.manifold.inner(x_t, diff, diff).mean()

        n_pred = self._norm(v_pred, x_t).unsqueeze(-1)
        n_target = self._norm(v_target, x_t).unsqueeze(-1)

        v_pred_n = v_pred / (n_pred + 1e-8)
        v_target_n = v_target / (n_target + 1e-8)

        if self.m_type == "euclidean":
            alignment = (v_pred_n * v_target_n).sum(dim=-1).mean()
        else:
            alignment = self.manifold.inner(x_t, v_pred_n, v_target_n).mean()

        return loss, alignment


    def kl_divergence(
        self, x: ManifoldTensor, mu: ManifoldTensor, p: torch.Tensor, eps: float = 1e-8
    ) -> float:
        """
        Compute the Kullback-Leibler divergence between the proportion of samples
        mapped to each Gaussian and the true distribution.

        Args:
            x: coordinates of the transported samples on the manifold (N x d).
            mu: coordinates of the Gaussian centers on the manifold (K x d).
            p: importance weights of each Gaussian.
            eps: small offset for stabilising KLD computation.

        Returns:
            kld: the KL-Divergence.
        """

        # expand dimensions for broadcasting
        x_exp = x.unsqueeze(1)  # Shape: (N, 1, d)
        mu_exp = mu.unsqueeze(0)  # Shape: (1, k, d)

        # compute the true pairwise hyperbolic distances directly
        if self.cross_curvature:
            distances = self.scaled_dist(x_exp, mu_exp) #normalizing for better cross-curvature comparison
        else:
            distances = self.manifold.dist(x_exp, mu_exp)  # Shape: (N, k)

        # find the index of the closest mu for each x
        nearest_gaussian = torch.argmin(distances, dim=1)

        # compute proportions
        transported_p = torch.zeros_like(p)
        idx, counts = nearest_gaussian.unique(return_counts=True)
        transported_p[idx] = counts / counts.sum().float()

        transported_p += eps
        transported_p = transported_p / transported_p.sum()

        # compute KL divergence
        kld = torch.nn.functional.kl_div(
            input=torch.log(transported_p),
            target=p,
            reduction="sum",
        )

        return kld


    def calculate_all(self, pred, target, mode="sample", step=0, x_t=None):
        results = {}

        if mode == "vector":
            assert x_t is not None, "x_t required for RFM metric"

            error, align = self.calculate_rfm_loss(pred, target, x_t)
            norm_stats = self.calculate_vector_norm_stats(pred, x_t)
            tangent_stats = self.calculate_tangency_violation(x_t, pred)

            results["val_vec/rfm_loss"] = error
            results["val_vec/alignment"] = align
            results["val_vec/finite_fraction"] = self.finite_fraction(pred)
            results["val_vec/norm_mean"] = norm_stats["mean"]
            results["val_vec/norm_max"] = norm_stats["max"]
            results["val_vec/norm_std"] = norm_stats["std"]
            results["val_vec/norm_p95"] = norm_stats["p95"]
            results["val_vec/tangency_violation_abs"] = tangent_stats["absolute"]
            results["val_vec/tangency_violation_rel"] = tangent_stats["relative"]

        elif mode == "sample":
            if "sinkhorn_knopp" in self.active_metrics:
                results["val_sample/sinkhorn_knopp"] = self.calculate_sinkhorn_divergence(
                    pred, target
                )

            if "mmd" in self.active_metrics:
                results["val_sample/mmd"] = self.calculate_mmd(pred, target)

            if "epsilon_coverage" in self.active_metrics:
                results["val_sample/epsilon_coverage"] = (
                    self.calculate_epsilon_coverage(pred, target)
                )

            if "epsilon_precision" in self.active_metrics:
                results["val_sample/epsilon_precision"] = (
                    self.calculate_epsilon_precision(pred, target)
                )

            if "frechet_variance" in self.active_metrics:
                frechet_pred = self.calculate_frechet_variance(pred)
                frechet_target = self.calculate_frechet_variance(target)

                results["val_sample/frechet_variance_pred"] = frechet_pred
                results["val_sample/frechet_variance_target"] = frechet_target
                results["val_sample/frechet_variance_ratio"] = (
                    frechet_pred / torch.clamp(frechet_target, min=1e-8)
                )

            if "dispersion" in self.active_metrics:
                disp_pred = self.calculate_dispersion(pred)
                disp_target = self.calculate_dispersion(target)

                results["val_sample/dispersion_predicted"] = disp_pred
                results["val_sample/dispersion_target"] = disp_target
                results["val_sample/dispersion_ratio"] = disp_pred / torch.clamp(
                    disp_target, min=1e-8
                )

            #! Should add radial/angular decomposition here

            if "stability" in self.active_metrics:
                results["val_sample/finite_fraction"] = self.finite_fraction(pred)

            if pred.shape == target.shape:
                if self.m_type == "euclidean":
                    dist_val = torch.linalg.norm(pred - target, dim=-1).mean()
                else:
                    if self.cross_curvature:
                        dist_val = self.scaled_dist(pred, target).mean() #normalizing for better cross-curvature comparison
                    else:
                        dist_val = self.manifold.dist(pred, target).mean()
                    

                results["val_sample/geodesic_dist"] = dist_val
            if self.cfg.get("save_densities", False):
                self.save_density_state(pred, target, step)

        return results


    def save_density_state(
        self,
        x_gen,
        x_real,
        step,
    ):
        path = "results/densities"
        os.makedirs(path, exist_ok=True)

        manifold_name = self.manifold.__class__.__name__

        data = {
            "x_gen": x_gen.detach().cpu(),
            "x_real": x_real.detach().cpu(),
            "step": step,
            "manifold": manifold_name,
            "manifold_type": self.m_type,
            "curvature_backend": self.curvature,
            "kappa": self.kappa,
            "origin": None if self.origin is None else self.origin,
            "num_samples": x_gen.shape[0],
            "cfg": self.cfg,
        }

        torch.save(data, f"{path}/{manifold_name}_step_{step:06d}.pt")


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
