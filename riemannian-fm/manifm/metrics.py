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
        gcfg = cfg.get("general", None)
        self.gcfg = gcfg

        self.metrics_used = cfg.get(
            "metrics_used",
            {
                "sinkhorn_knopp": True,
                "tangent_sinkhorn_knopp": True,
                "mmd": True,
                "tangent_mmd": True,
                "epsilon_coverage": True,
                "epsilon_precision": True,
                "tangent_epsilon_coverage": True,
                "tangent_epsilon_precision": True,
                "frechet_variance": True,
                "tangent_frechet_variance": True,
                "dispersion": True,
                "tangent_dispersion": True,
                "radial": True,
                "stability": True,
                "rfm": True,
                "volume_scaling": True,
                "snr": True,
            },
        )
        self.metrics_params = cfg.get("metrics_param", {}) or {} # parameters for metrics, e.g. blur for Sinkhorn-Knopp
        self.cross_curvature = cfg.get("cross_curvature", False) # whether to normalize metrics for better cross-curvature comparison

        self.m_type = gcfg.get("manifold", "euclidean").lower()
        self.dim = int(gcfg.get("dim"))
        self.curvature = gcfg.get("curvature", 1.0)
                                        # origin is North pole for spherical,
                                        # plain origin for Euclidean and hyperbolic
        self.origin = gcfg.get("origin", None)

        if self.m_type == "euclidean":
            self.manifold = Euclidean()
            self.kappa = 0.0            # flat
        elif self.m_type == "sphere":
            self.manifold = SphereCurvature(c = self.curvature)
            self.kappa = self.curvature # positive curvature (spherical)
        elif self.m_type == "poincare":
            self.manifold = PoincareBall(c = self.curvature)
                                        # negative curvature (hyperbolic)
            self.kappa = -self.curvature
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

        if self.m_type == "sphere":
            radius = 1.0 / torch.sqrt(
                torch.as_tensor(self.curvature, device=x.device, dtype=x.dtype)
            )
            origin = torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
            origin[:, 0] = radius  #should we use freichet mean here instead? or does it not matter as long as it's fixed? should be fine as long as it's fixed.
            return origin

        raise ValueError(f"origin not defined for manifold: {self.m_type}")
    

    def tangent_coordinates(self, x):
        """ map samples x on the manifold to the tangent space
        at the fixed origin using the logarithmic map """
        origin = self.get_origin(x).expand_as(x)

        if self.m_type == "euclidean":
            return x - origin
        return self.manifold.logmap(origin, x)


    def tangent_coordinates_scaled(self, x):
        """ scale the tangent coordinates by sqrt(|kappa|)
        only for tangent-space Sinkhorn cross-curvature comparison """
        v = self.tangent_coordinates(x)

        normalize = self.metrics_params.get("normalize_tangent_sinkhorn", True)
        if normalize and self.kappa != 0:
            v = v * torch.sqrt(
                torch.as_tensor(abs(self.kappa), device=v.device, dtype=v.dtype)
            )
        return v
    

    def calculate_tangent_sinkhorn(
        self,
        x_gen,
        x_real,
        p = 1,
        blur = None,
    ):
        """ Sinkhorn divergence after projecting both distributions
        to the tangent space at the fixed origin via the logarithmic map """
        if blur is None:
            blur = self.metrics_params.get("sinkhorn_blur", 0.05)

        v_gen = self.tangent_coordinates_scaled(x_gen)
        v_real = self.tangent_coordinates_scaled(x_real)

        solver = SamplesLoss(
            loss = "sinkhorn",
            p = p,
            blur = blur,
            debias = True,
            backend = "tensorized",
        )
        val = solver(v_gen, v_real)
        val = torch.clamp(val, min = 0.0) ** (1.0 / p)
        return val
    

    def calculate_with_snr(self, metric_fn, x_gen, x_real, n_boot=30, frac=0.8, **kwargs):
        """
        Estimate mean, std and signal-to-noise ratio (SNR = mean/std) of a
        metric by resampling (without replacement) sub-batches of the
        evaluation set.
        """
        n_gen = x_gen.shape[0]
        n_real = x_real.shape[0]
        sub_n_gen = max(2, int(frac * n_gen))
        sub_n_real = max(2, int(frac * n_real))

        vals = []
        for _ in range(n_boot):
            idx_g = torch.randperm(n_gen, device=x_gen.device)[:sub_n_gen]
            idx_r = torch.randperm(n_real, device=x_real.device)[:sub_n_real]
            with torch.no_grad():
                v = metric_fn(x_gen[idx_g], x_real[idx_r], **kwargs)
            vals.append(v.detach())

        vals = torch.stack(vals)
        mean = vals.mean()
        std = vals.std(unbiased=True)
        snr = mean / torch.clamp(std, min=1e-8)
        return {"mean": mean, "std": std, "snr": snr}


    def scaled_dist(self, x, y):
        if self.m_type == "euclidean":
            return torch.cdist(x, y, p=2)

        d = self.manifold.dist(x, y)

        if self.kappa != 0:
            scale = torch.sqrt(torch.tensor(abs(self.kappa), device=d.device, dtype=d.dtype))
            d = d * scale

        return d


    def calculate_sinkhorn_divergence(
        self,
        x_gen,
        x_real,
        p = 1,
        blur = None,
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
            blur = self.metrics_params.get("sinkhorn_blur", 0.05)

        # REMOVED: blur = blur * (abs(self.kappa) ** 0.5) 
        # The geodesic_cost already scales the distance, meaning the cost matrix is invariant. 
        # Scaling blur would result in curvature-dependent entropic regularization.
        # if self.cross_curvature and self.kappa != 0: #normalizing for better cross-curvature comparison
        #     blur = blur * (abs(self.kappa) ** 0.5)
            
        if self.m_type == "euclidean":  # debias = True, i.e. use Sinkhorn divergence
            solver = SamplesLoss(
                loss = "sinkhorn",
                p = p,
                blur = blur,
                debias = True,
                backend = "tensorized",
            )
            val = solver(x_gen, x_real)
            
        else:

            def geodesic_cost(x, y):
                try:
                    # do an explicit pairwise comparison
                    x_exp = x.unsqueeze(1)  # (N, 1, D)
                    y_exp = y.unsqueeze(0)  # (1, M, D)
                    
                    # if self.cross_curvature:
                    #    d = self.scaled_dist(x_exp, y_exp) #normalizing for better cross-curvature comparison
                    #    return d ** p
                    # else:
                    return self.manifold.dist(x_exp, y_exp) ** p

                except:
                    n = x.shape[0]
                    m = y.shape[0]
                    cost_matrix = torch.zeros((n, m), device=x.device, dtype=x.dtype)
                    y_exp = y.unsqueeze(0)
                    chunk_size = 256
                    
                    for i in range(0, n, chunk_size):
                        x_chunk = x[i : i + chunk_size].unsqueeze(1)
                        cost_matrix[i : i + chunk_size] = self.manifold.dist(x_chunk, y_exp) ** p
                        
                    return cost_matrix


            solver = SamplesLoss(
                loss = "sinkhorn",
                p = p,
                blur = blur,
                debias = True, # debiasing so we get the Sinkhorn divergence by Feydy et al. (2019)
                cost = geodesic_cost,
                backend = "tensorized",
            )
            val = solver(x_gen, x_real)

        val = torch.clamp(val, min=0.0) ** (1.0 / p)
        return val

    #! Add decomposed Sinkhorn Knopp for radial and angular


    def pairwise_dist(self, x, y, chunk_size=256):
        if self.m_type == "euclidean":
            return torch.cdist(x, y, p=2)

        try:
            x_exp = x.unsqueeze(1)
            y_exp = y.unsqueeze(0)
            
            if self.cross_curvature:
                return self.scaled_dist(x_exp, y_exp) #normalizing for better cross-curvature comparison
            else:
                return self.manifold.dist(x_exp, y_exp)

        except:
            n = x.shape[0]
            m = y.shape[0]
            dist_matrix = torch.zeros((n, m), device=x.device, dtype=x.dtype)
            
            y_exp = y.unsqueeze(0)  # Shape: (1, M, D)
            
            for i in range(0, n, chunk_size):
                x_chunk = x[i : i + chunk_size].unsqueeze(1)  # Shape: (chunk, 1, D)
                
                if self.cross_curvature:
                    d_chunk = self.scaled_dist(x_chunk, y_exp)
                else:
                    d_chunk = self.manifold.dist(x_chunk, y_exp)
                    
                dist_matrix[i : i + chunk_size] = d_chunk
                
            return dist_matrix


    def _mmd_from_distances(self, dxx, dyy, dxy, sigma=None):
        """ shared RBF-kernel MMD computation given precomputed pairwise
        distance matrices; used by both the geodesic (D_r) and
        tangent-space (D_e) MMD variants below """
        if sigma is None:
            vals = dxy.detach().flatten()
            vals = vals[vals > 0]
            sigma = torch.median(vals)
            sigma = torch.clamp(sigma, min=1e-6)

        kxx = torch.exp(-(dxx**2) / (2 * sigma**2))
        kyy = torch.exp(-(dyy**2) / (2 * sigma**2))
        kxy = torch.exp(-(dxy**2) / (2 * sigma**2))

        return kxx.mean() + kyy.mean() - 2 * kxy.mean()


    def calculate_mmd(self, x_gen, x_real, sigma=None):
        """
        D_r: MMD with geodesic RBF kernel (pairwise geodesic distances)
        """
        dxx = self.pairwise_dist(x_gen, x_gen)
        dyy = self.pairwise_dist(x_real, x_real)
        dxy = self.pairwise_dist(x_gen, x_real)

        return self._mmd_from_distances(dxx, dyy, dxy, sigma)


    def calculate_tangent_mmd(self, x_gen, x_real, sigma=None):
        """
        D_e: MMD with Euclidean RBF kernel, computed after projecting both
        distributions to the tangent space at the fixed origin via the
        logarithmic map (same projection used by calculate_tangent_sinkhorn).
        """
        v_gen = self.tangent_coordinates_scaled(x_gen)
        v_real = self.tangent_coordinates_scaled(x_real)

        dxx = torch.cdist(v_gen, v_gen, p=2)
        dyy = torch.cdist(v_real, v_real, p=2)
        dxy = torch.cdist(v_gen, v_real, p=2)

        return self._mmd_from_distances(dxx, dyy, dxy, sigma)


    def _epsilon_ball_stat(
        self,
        d_cross,
        d_self_ref,
        eps=None,
        eps_multiplier=None,
        return_eps=False,
    ):
        """
        Shared epsilon-ball coverage/precision computation given precomputed
        distance matrices. d_cross: distances used for the nearest-neighbour
        test (rows = query set). d_self_ref: within-real-set distances used
        to auto-derive eps from the median nearest-neighbour distance, if eps
        is not given explicitly. Used by both the geodesic (D_r) and
        tangent-space (D_e) coverage/precision variants below.
        """
        if eps_multiplier is None:
            eps_multiplier = self.metrics_params.get("coverage_eps_multiplier", 1.0)

        if eps is None:
            n = d_self_ref.shape[0]
            d_self_ref = d_self_ref + torch.eye(n, device=d_self_ref.device) * 1e9
            eps = d_self_ref.min(dim=1).values.median()
            eps = eps_multiplier * eps

        nearest = d_cross.min(dim=1).values
        stat = (nearest <= eps).float().mean()

        if return_eps:
            return stat, eps
        return stat


    def calculate_epsilon_coverage(
        self,
        x_gen,
        x_real,
        eps=None,
        eps_multiplier=None,
        return_eps=False,
    ):
        """
        D_r: measures how much of the target support is reached by generated
        samples, using geodesic distances.
        """
        d_cross = self.pairwise_dist(x_real, x_gen)
        d_self_ref = self.pairwise_dist(x_real, x_real)
        return self._epsilon_ball_stat(d_cross, d_self_ref, eps, eps_multiplier, return_eps)


    def calculate_epsilon_precision(
        self,
        x_gen,
        x_real,
        eps=None,
        eps_multiplier=None,
        return_eps=False,
    ):
        """
        D_r: measures how many generated samples lie within epsilon of the
        target support, using geodesic distances.
        """
        d_cross = self.pairwise_dist(x_gen, x_real)
        d_self_ref = self.pairwise_dist(x_real, x_real)
        return self._epsilon_ball_stat(d_cross, d_self_ref, eps, eps_multiplier, return_eps)


    def calculate_tangent_epsilon_coverage(
        self,
        x_gen,
        x_real,
        eps=None,
        eps_multiplier=None,
        return_eps=False,
    ):
        """
        D_e: same as calculate_epsilon_coverage but computed in tangent-space
        (Euclidean) coordinates instead of geodesic distances.
        """
        v_gen = self.tangent_coordinates_scaled(x_gen)
        v_real = self.tangent_coordinates_scaled(x_real)
        d_cross = torch.cdist(v_real, v_gen, p=2)
        d_self_ref = torch.cdist(v_real, v_real, p=2)
        return self._epsilon_ball_stat(d_cross, d_self_ref, eps, eps_multiplier, return_eps)


    def calculate_tangent_epsilon_precision(
        self,
        x_gen,
        x_real,
        eps=None,
        eps_multiplier=None,
        return_eps=False,
    ):
        """
        D_e: same as calculate_epsilon_precision but computed in tangent-space
        (Euclidean) coordinates instead of geodesic distances.
        """
        v_gen = self.tangent_coordinates_scaled(x_gen)
        v_real = self.tangent_coordinates_scaled(x_real)
        d_cross = torch.cdist(v_gen, v_real, p=2)
        d_self_ref = torch.cdist(v_real, v_real, p=2)
        return self._epsilon_ball_stat(d_cross, d_self_ref, eps, eps_multiplier, return_eps)


    def _norm(self, v, x):
        """
        Computes tangent-vector norms using the Riemannian metric
        """
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
    

    def elementwise_scaled_dist(self, x, y):
        if self.m_type == "euclidean":
            return torch.linalg.norm(x - y, dim = -1)

        d = self.manifold.dist(x, y)
        if self.kappa != 0:
            scale = torch.sqrt(torch.tensor(abs(self.kappa), device = d.device, dtype = d.dtype))
            d = d * scale
        return d


    def calculate_frechet_variance(self, samples):
        """
        D_r: Calculate Frechet variance using Frechet mean (geodesic)
        """
        mu = self.frechet_mean(samples)
        mu_expanded = mu.expand_as(samples)

        if self.cross_curvature:
            return self.elementwise_scaled_dist(samples, mu_expanded).pow(2).mean()
        elif self.m_type == "euclidean":
            return torch.linalg.norm(samples - mu_expanded, dim=-1).pow(2).mean()
        else:
            return self.manifold.dist(samples, mu_expanded).pow(2).mean()
        
        # return self.manifold.dist(samples, mu_expanded).pow(2).mean()
        # not using this as, we would need to normalize logmap, for statistics not needed.
        #return self.scaled_dist(samples, mu_expanded).pow(2).mean() #normalizing for better cross-curvature comparison  


    def calculate_tangent_frechet_variance(self, samples):
        """
        D_e: variance of samples around their mean in tangent-space
        coordinates (tangent_coordinates_scaled), the Euclidean counterpart
        to calculate_frechet_variance. Since the tangent space is flat, the
        mean there is just the arithmetic mean -- no iterative Frechet-mean
        search is needed, unlike the manifold case in frechet_mean().
        """
        v = self.tangent_coordinates_scaled(samples)
        mu = v.mean(dim=0, keepdim=True)
        return (v - mu).pow(2).sum(dim=-1).mean()


    def calculate_dispersion(self, samples):
        """
        D_r: pairwise dispersion calculation (geodesic), simpler version of Frechet variance
        """
        d = self.pairwise_dist(samples, samples)
        n = samples.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=samples.device)
        return d[mask].mean()


    def calculate_tangent_dispersion(self, samples):
        """
        D_e: pairwise dispersion in tangent-space coordinates (same projection
        used by calculate_tangent_sinkhorn / calculate_tangent_mmd), the
        Euclidean counterpart to calculate_dispersion.
        """
        v = self.tangent_coordinates_scaled(samples)
        d = torch.cdist(v, v, p=2)
        n = v.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=v.device)
        return d[mask].mean()


    def calculate_dispersion_ratio(self, x_pred, x_target):
        """
        Diversity: Calculate dispersion ratio of predicted with target dispersion (D_r)
        """
        gen_disp = self.calculate_dispersion(x_pred)
        real_disp = self.calculate_dispersion(x_target)
        return gen_disp / torch.clamp(real_disp, min=1e-8)


    def calculate_tangent_dispersion_ratio(self, x_pred, x_target):
        """
        Diversity: same as calculate_dispersion_ratio but using tangent-space
        (D_e) dispersion instead of geodesic (D_r) dispersion.
        """
        gen_disp = self.calculate_tangent_dispersion(x_pred)
        real_disp = self.calculate_tangent_dispersion(x_target)
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

            # ADDED: curvature normalisation.
            # Vector fields scale with R. Squared error scales with R^2.
            # Multiply by |K| (which is 1/R^2) to normalize.
            # if self.cross_curvature and self.kappa != 0:
            #     loss = loss * abs(self.kappa)

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
    

    def intrinsic_dim(self):
        """ intrinsic dim of manifold, i.e. just - 1;
        needed because for the sphere implementation,
        the coordinates are ambient instead of intrinsic,
        whereas for the geometric theory, we need intrinsic """
        if self.m_type == "sphere":
            return self.dim - 1         # since spherical impl. is in ambient space
        return self.dim
    

    def radial_geodesic_distance(self, x):
        """ geodesic radius from origin, elementwise """
        origin = self.get_origin(x).expand_as(x)
        if self.m_type == "euclidean":
            return torch.linalg.norm(x - origin, dim=-1)
                                        # returns shape [N]
        return self.manifold.dist(origin, x)
    

    def radial_jacobian_ratio(self, r, eps = 1e-8):
        """ compute S_kapper(r) / r ratio:
        - kappa > 0 : sin(sqrt(kappa) r) / (sqrt(kappa) r)
        - kappa = 0 : 1
        - kappa < 0 : sinh(sqrt(-kappa) r) / (sqrt(-kappa) r)
        """
        if self.kappa == 0:             # trivial case
            return torch.ones_like(r)
                                        # take the absolutus for hyperbolic case below
        abs_kappa = torch.as_tensor(abs(self.kappa), device = r.device, dtype = r.dtype)
        rho = torch.sqrt(abs_kappa) * r
                                        # avoid zero-division using eps argument
        rho_safe = torch.clamp(rho, min = eps)
        if self.kappa > 0:              # --> spherical
            ratio = torch.sin(rho_safe) / rho_safe
                                        # exact limit at rho = 0
            ratio = torch.where(rho < 1e-6, torch.ones_like(ratio), ratio)
                                        # just for numerical safety near the antipode
            ratio = torch.clamp(ratio, min = 0.0)
        else:                           # --> hyperbolic
            ratio = torch.sinh(rho_safe) / rho_safe
                                        # exact limit at rho = 0
            ratio = torch.where(rho < 1e-6, torch.ones_like(ratio), ratio)
        return ratio


    def volume_scaling_values(self, x):
        """ compute J_kappa(r) = (S_kappa(r) / r)^(d-1) for a point x """
        d_intrinsic = self.intrinsic_dim()
        exponent = d_intrinsic - 1      # this metric is intrinsic, rather than ambient

                                        # if intr. dim. is 1, there are no angulardirections
                                        # so the Jacobian factor is just 1's
        if exponent == 0:
            return torch.ones(x.shape[0], device = x.device, dtype = x.dtype)

        r = self.radial_geodesic_distance(x)
        ratio = self.radial_jacobian_ratio(r)
        return ratio.pow(exponent)


    def calculate_volume_scaling_mean(self, samples):
        """ mean (expected) "experienced" volume scaling """
        J = self.volume_scaling_values(samples)
        return J.mean()


    def calculate_volume_scaling_variance(self, samples):
        """ variance of "experienced" volume scaling """
        J = self.volume_scaling_values(samples)
        return J.var(unbiased = False)


    def calculate_volume_scaling_gap(self, start_samples, target_samples):
        """ start-target volume-growth gap, for p0 and p1, resp. """
        mean_start = self.calculate_volume_scaling_mean(start_samples)
        mean_target = self.calculate_volume_scaling_mean(target_samples)
        return mean_target - mean_start
    

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


    def calculate_all(self, pred, target, mode = "sample", step = 0, x_t = None, start = None):
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
            if self.metrics_used.get("sinkhorn_knopp", False):
                sinkhorn_val = self.calculate_sinkhorn_divergence(pred, target)
                results["val_sample/sinkhorn_knopp"] = sinkhorn_val

            if self.metrics_used.get("tangent_sinkhorn_knopp", False):
                tangent_sinkhorn_val = self.calculate_tangent_sinkhorn(pred, target)
                results["val_sample/tangent_sinkhorn_knopp"] = tangent_sinkhorn_val
                if self.metrics_used.get("sinkhorn_knopp", False):
                    results["val_sample/tangent_to_geodesic_sinkhorn_ratio"] = (
                        tangent_sinkhorn_val / torch.clamp(sinkhorn_val, min = 1e-8)
                    )

            mmd_val = None
            if self.metrics_used.get("mmd", False):
                mmd_val = self.calculate_mmd(pred, target)
                results["val_sample/mmd"] = mmd_val

            if self.metrics_used.get("tangent_mmd", False):
                tangent_mmd_val = self.calculate_tangent_mmd(pred, target)
                results["val_sample/tangent_mmd"] = tangent_mmd_val
                if mmd_val is not None:
                    results["val_sample/tangent_to_geodesic_mmd_ratio"] = (
                        tangent_mmd_val / torch.clamp(mmd_val, min = 1e-8)
                    )

            if self.metrics_used.get("epsilon_coverage", False):
                coverage, eps_cov = self.calculate_epsilon_coverage(
                    pred, target, return_eps = True
                )
                results["val_sample/epsilon_coverage"] = coverage
                results["val_sample/epsilon_coverage_eps"] = eps_cov

            if self.metrics_used.get("epsilon_precision", False):
                precision, eps_prec = self.calculate_epsilon_precision(
                    pred, target, return_eps = True
                )
                results["val_sample/epsilon_precision"] = precision
                results["val_sample/epsilon_precision_eps"] = eps_prec

            if self.metrics_used.get("tangent_epsilon_coverage", False):
                tangent_coverage, tangent_eps_cov = self.calculate_tangent_epsilon_coverage(
                    pred, target, return_eps = True
                )
                results["val_sample/tangent_epsilon_coverage"] = tangent_coverage
                results["val_sample/tangent_epsilon_coverage_eps"] = tangent_eps_cov

            if self.metrics_used.get("tangent_epsilon_precision", False):
                tangent_precision, tangent_eps_prec = self.calculate_tangent_epsilon_precision(
                    pred, target, return_eps = True
                )
                results["val_sample/tangent_epsilon_precision"] = tangent_precision
                results["val_sample/tangent_epsilon_precision_eps"] = tangent_eps_prec

            if self.metrics_used.get("snr", False):
                n_boot = self.metrics_params.get("snr_n_boot", 30)
                frac = self.metrics_params.get("snr_frac", 0.8)

                snr_sinkhorn_r = snr_sinkhorn_e = None
                if self.metrics_used.get("sinkhorn_knopp", False):
                    snr_sinkhorn_r = self.calculate_with_snr(
                        self.calculate_sinkhorn_divergence, pred, target,
                        n_boot = n_boot, frac = frac,
                    )
                    results["val_sample/snr_sinkhorn_r_mean"] = snr_sinkhorn_r["mean"]
                    results["val_sample/snr_sinkhorn_r_std"] = snr_sinkhorn_r["std"]
                    results["val_sample/snr_sinkhorn_r"] = snr_sinkhorn_r["snr"]

                if self.metrics_used.get("tangent_sinkhorn_knopp", False):
                    snr_sinkhorn_e = self.calculate_with_snr(
                        self.calculate_tangent_sinkhorn, pred, target,
                        n_boot = n_boot, frac = frac,
                    )
                    results["val_sample/snr_sinkhorn_e_mean"] = snr_sinkhorn_e["mean"]
                    results["val_sample/snr_sinkhorn_e_std"] = snr_sinkhorn_e["std"]
                    results["val_sample/snr_sinkhorn_e"] = snr_sinkhorn_e["snr"]

                if snr_sinkhorn_r is not None and snr_sinkhorn_e is not None:
                    results["val_sample/snr_ratio_sinkhorn_r_over_e"] = (
                        snr_sinkhorn_r["snr"] / torch.clamp(snr_sinkhorn_e["snr"], min = 1e-8)
                    )

                snr_mmd_r = snr_mmd_e = None
                if self.metrics_used.get("mmd", False):
                    snr_mmd_r = self.calculate_with_snr(
                        self.calculate_mmd, pred, target,
                        n_boot = n_boot, frac = frac,
                    )
                    results["val_sample/snr_mmd_r_mean"] = snr_mmd_r["mean"]
                    results["val_sample/snr_mmd_r_std"] = snr_mmd_r["std"]
                    results["val_sample/snr_mmd_r"] = snr_mmd_r["snr"]

                if self.metrics_used.get("tangent_mmd", False):
                    snr_mmd_e = self.calculate_with_snr(
                        self.calculate_tangent_mmd, pred, target,
                        n_boot = n_boot, frac = frac,
                    )
                    results["val_sample/snr_mmd_e_mean"] = snr_mmd_e["mean"]
                    results["val_sample/snr_mmd_e_std"] = snr_mmd_e["std"]
                    results["val_sample/snr_mmd_e"] = snr_mmd_e["snr"]

                if snr_mmd_r is not None and snr_mmd_e is not None:
                    results["val_sample/snr_ratio_mmd_r_over_e"] = (
                        snr_mmd_r["snr"] / torch.clamp(snr_mmd_e["snr"], min = 1e-8)
                    )

            if self.metrics_used.get("frechet_variance", False):
                frechet_pred = self.calculate_frechet_variance(pred)
                frechet_target = self.calculate_frechet_variance(target)

                results["val_sample/frechet_variance_pred"] = frechet_pred
                results["val_sample/frechet_variance_target"] = frechet_target
                results["val_sample/frechet_variance_ratio"] = (
                    frechet_pred / torch.clamp(frechet_target, min=1e-8)
                )

            if self.metrics_used.get("tangent_frechet_variance", False):
                tangent_frechet_pred = self.calculate_tangent_frechet_variance(pred)
                tangent_frechet_target = self.calculate_tangent_frechet_variance(target)

                results["val_sample/tangent_frechet_variance_pred"] = tangent_frechet_pred
                results["val_sample/tangent_frechet_variance_target"] = tangent_frechet_target
                results["val_sample/tangent_frechet_variance_ratio"] = (
                    tangent_frechet_pred / torch.clamp(tangent_frechet_target, min=1e-8)
                )

            if self.metrics_used.get("dispersion", False):
                disp_pred = self.calculate_dispersion(pred)
                disp_target = self.calculate_dispersion(target)

                results["val_sample/dispersion_predicted"] = disp_pred
                results["val_sample/dispersion_target"] = disp_target
                results["val_sample/dispersion_ratio"] = disp_pred / torch.clamp(
                    disp_target, min = 1e-8
                )

            if self.metrics_used.get("tangent_dispersion", False):
                tangent_disp_pred = self.calculate_tangent_dispersion(pred)
                tangent_disp_target = self.calculate_tangent_dispersion(target)

                results["val_sample/tangent_dispersion_predicted"] = tangent_disp_pred
                results["val_sample/tangent_dispersion_target"] = tangent_disp_target
                results["val_sample/tangent_dispersion_ratio"] = tangent_disp_pred / torch.clamp(
                    tangent_disp_target, min = 1e-8
                )

            if self.metrics_used.get("volume_scaling", False):
                mean_pred = self.calculate_volume_scaling_mean(pred)
                mean_target = self.calculate_volume_scaling_mean(target)

                var_pred = self.calculate_volume_scaling_variance(pred)
                var_target = self.calculate_volume_scaling_variance(target)

                results["val_sample/volume_scaling_mean_pred"] = mean_pred
                results["val_sample/volume_scaling_mean_target"] = mean_target
                results["val_sample/volume_scaling_variance_pred"] = var_pred
                results["val_sample/volume_scaling_variance_target"] = var_target
                                        # to see whether experienced volume growth differs between gt and pred
                results["val_sample/volume_scaling_gap_target_minus_pred"] = mean_target - mean_pred

                if start is not None:
                    mean_start = self.calculate_volume_scaling_mean(start)
                    var_start = self.calculate_volume_scaling_variance(start)
                    gap_start_target = self.calculate_volume_scaling_gap(start, target)

                    results["val_sample/volume_scaling_mean_start"] = mean_start
                    results["val_sample/volume_scaling_variance_start"] = var_start
                    results["val_sample/volume_scaling_gap_start_target"] = gap_start_target

            #! Should add radial/angular decomposition here

            if self.metrics_used.get("stability", False):
                results["val_sample/finite_fraction"] = self.finite_fraction(pred)

            if pred.shape == target.shape:
                if self.m_type == "euclidean":
                    dist_val = torch.linalg.norm(pred - target, dim = -1).mean()
                else:
                    if self.cross_curvature:
                        dist_val = self.scaled_dist(pred, target).mean() #normalizing for better cross-curvature comparison
                    else:
                        dist_val = self.manifold.dist(pred, target).mean()
                results["val_sample/geodesic_dist"] = dist_val

            if self.cfg.get("save_densities", False):
                self.save_density_state(pred, target, step)

        return results