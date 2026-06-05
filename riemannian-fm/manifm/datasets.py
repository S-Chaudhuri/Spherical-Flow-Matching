"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import json
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from manifm.manifolds import PoincareBall, Euclidean, SphereCurvature


class CheckerboardDataset(Dataset):
    """
    defines the checkerboard distribution on the square [-1,1]^2, on the black squares.
    """

    def __init__(
        self, manifold, manifold_name, dim, n_samples=20000, num_square=4, tangent_scale=0.7
    ):
        self.manifold = manifold
        self.manifold_name = manifold_name
        self.dim = dim
        self.n_samples = n_samples
        self.num_square = num_square
        self.tangent_scale = tangent_scale

    def __len__(self):
        return self.n_samples

    def _checkerboard2d(self):

        x = torch.rand(1, 2) * self.num_square

        cell_x = torch.floor(x[:, 0]).to(torch.int)
        cell_y = torch.floor(x[:, 1]).to(torch.int)
        is_white = (cell_x + cell_y) % 2 == 0

        can_shift_right = cell_x < self.num_square - 1
        x[:, 0] = x[:, 0] + (is_white & can_shift_right).float()
        can_shift_down = cell_y < self.num_square - 1
        x[:, 1] = x[:, 1] + (is_white & ~can_shift_right & can_shift_down).float()

        is_corner = is_white & ~can_shift_right & ~can_shift_down
        x[:, 0] = x[:, 0] - (self.num_square - 1) * is_corner.float()

        return (x / self.num_square * 2 - 1).float().squeeze(0)


class GeneralDataset(Dataset):
    """
    General dataset for sampling pairs of points (x0, x1) on a specified manifold,
    where x0 and x1 are sampled from specified distributions (e.g., uniform, Gaussian).
    The possible manifolds include "sphere", "poincare", and "euclidean". The possible distributions: "gaussian".
    """

    def __init__(self, cfg):
        self.cfg = cfg
        gcfg = cfg.get("general", None)
        self.gcfg = gcfg

        self.dim = int(gcfg.dim)
        self.n_samples = int(gcfg.n_samples)

        self.x0_dist = gcfg.get("x0_dist", None)
        self.x1_dist = gcfg.get("x1_dist", None)
        self.std_x0 = gcfg.get("std_x0", None)
        self.std_x1 = gcfg.get("std_x1", None)
        self.mean_x0 = gcfg.get("mean_x0", None)
        self.mean_x1 = gcfg.get("mean_x1", None)
        self.radius_x0 = gcfg.get("radius_x0", None)    # radius for Gaussian ring
        self.radius_x1 = gcfg.get("radius_x1", None)    # radius for Gaussian ring

        # saving configuration for evalutaion
        self.eval_n_pairs = int(cfg.get("eval_n_pairs", 100))
        self.eval_t_values = cfg.get("eval_t_values", None)
        self.save_artifacts = bool(cfg.get("save_artifacts", True))

        # --- Manifold ---
        self.manifold_name = gcfg.manifold
        self.curvature = float(gcfg.get("curvature", 1.0))

        if self.manifold_name == "sphere":
            self.manifold = SphereCurvature(c=self.curvature)
        elif self.manifold_name == "poincare":
            self.manifold = PoincareBall(c=self.curvature)
        elif self.manifold_name == "euclidean":
            self.manifold = Euclidean()
        else:
            raise ValueError("unknown manifold")

        self._checkerboard = None
        if (
            self._dist_key(self.x1_dist) == "checkerboard"
            or self._dist_key(self.x0_dist) == "checkerboard"
        ):
            self._checkerboard = CheckerboardDataset(
                manifold=self.manifold,
                manifold_name=self.manifold_name,
                dim=self.dim,
                n_samples=self.n_samples,
                num_square=int(self.cfg.get("checkerboard").get("num_square", None)),
                tangent_scale=float(self.cfg.get("checkerboard").get("tangent_scale", None)),
            )

        self.reference_origin = self.get_reference_origin()

        if self.save_artifacts:
            self._load_or_create_fixed_dataset()
        else:
            self.x0_all = None
            self.x1_all = None
            self.eval_x0 = None
            self.eval_x1 = None
            self.eval_t = None

    def check_point_on_manifold(self, mean, manifold, tol = 1e-5):
        """
        Validates that mean(s) lie on the correct manifold

        Supports:
        - single mean: shape (d,)
        - MoG means: shape (K, d) or list of vectors
        """

        if mean is None:
            return

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=torch.float32)

        if mean.ndim == 1:
            mean = mean.unsqueeze(0)

        norms = torch.norm(mean, dim=-1)

        if manifold == "poincare":
            if not torch.all(norms < 1.0):
                raise ValueError(f"Poincaré mean(s) must have norm < 1. Got norms: {norms}")

        elif manifold == "sphere":
            if not torch.all(torch.abs(norms - np.sqrt(1 / self.curvature)) < tol):
                raise ValueError(
                    f"Sphere mean(s) must have norm ≈ {np.sqrt(1 / self.curvature)}. Got norms: {norms}"
                )

        elif manifold == "euclidean":
            pass

        else:
            raise ValueError(f"Unknown manifold: {manifold}")

    def _to_tensor(self, x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().clone().float()
        return torch.tensor(x, dtype=torch.float32)

    def _dist_key(self, dist_name):
        if dist_name is None:
            return None
        return str(dist_name).lower()

    def get_reference_origin(self):
        """
        Fixed origin used for geodesic dilation:
        - custom general.origin if provided
        - otherwise zero for euclidean / poincare
        - north pole for sphere
        """
        origin = self.gcfg.get("origin", None)
        if origin is not None:
            origin = self._to_tensor(origin)
        elif self.manifold_name in ["euclidean", "poincare"]:
            origin = torch.zeros(self.dim, dtype=torch.float32)
        elif self.manifold_name == "sphere":
            origin = torch.zeros(self.dim, dtype=torch.float32)
            origin[0] = np.sqrt(1.0 / self.curvature)
        else:
            raise ValueError(f"unknown manifold: {self.manifold_name}")

        self.check_point_on_manifold(origin, self.manifold_name)
        return origin

    def normalize_tangent(self, z):
        if not self.gcfg.get("normalize_tangent_distributions", False):
            return z
        if self.manifold_name == "euclidean":
            return z
        curvature = abs(float(self.curvature))
        if curvature == 0.0:
            return z
        return z / (curvature ** 0.5)

    def tangent_to_manifold(self, z):
        single = False
        if z.ndim == 1:
            z = z.unsqueeze(0)
            single = True
        if self.manifold_name == "euclidean":
            x = z
        else:
            origin = self.reference_origin.to(device = z.device, dtype = z.dtype).view(1, -1)
            origin = origin.expand_as(z)

            if hasattr(self.manifold, "proju"):
                z = self.manifold.proju(origin, z)
            if self.manifold_name == "poincare":
                lambda_o = self.manifold.lambda_x(origin, keepdim =True)
                z = z / lambda_o
            x = self.manifold.expmap(origin, z)
            if hasattr(self.manifold, "projx"):
                x = self.manifold.projx(x)

        if single:
            return x.squeeze(0)
        return x


    def _sample_checkerboard_tangent(self):
        xy = self._checkerboard._checkerboard2d()
        xy = xy * self._checkerboard.tangent_scale
        z = torch.zeros(self.dim, dtype = xy.dtype, device = xy.device)

        if self.manifold_name == "sphere":
            if self.dim < 3:
                raise ValueError("sphere checkerboard requires ambient dim >= 3 for a 2D tangent plane")
            z[1:3] = xy
        else:
            z[:2] = xy
        return z


    def sample_tangent(self, dist_name, std = None, mean = None, radius = None):
        if std is None:
            std = 1.0
        if mean is None:
            mean = torch.zeros(self.dim, dtype = torch.float32)
        elif not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype = torch.float32)
        else:
            mean = mean.detach().clone().float()
        if not torch.is_tensor(std) and not isinstance(std, (float, int)):
            std = torch.tensor(std, dtype = torch.float32)

        dist_key = self._dist_key(dist_name)

        if dist_key == "gaussian":
            eps = torch.randn(self.dim, dtype = mean.dtype, device = mean.device) * float(std)
            return mean + eps
        elif dist_key == "gaussian-ring":
            if radius is None:
                raise ValueError("gaussian-ring requires radius")

            if torch.norm(mean) > 1e-8:
                raise ValueError(
                    "gaussian-ring is defined to be centered at the tangent-space origin;"
                    "set mean_x0/mean_x1 to null or zero"
                )

            direction = torch.randn(self.dim, dtype = mean.dtype, device = mean.device)

            if self.manifold_name == "sphere":
                direction[0] = 0.0

            direction = direction / torch.clamp(torch.norm(direction), min = 1e-8)
            radial_noise = torch.randn((), dtype = mean.dtype, device = mean.device) * float(std)
            r = float(radius) + radial_noise
            return r * direction

        elif dist_key == "mog":
            if mean.ndim != 2:
                raise ValueError("MoG mean must have shape (K, dim)")

            K = mean.shape[0]

            weights_cfg = self.gcfg.get("weights", None)
            if weights_cfg is None:
                weights = torch.ones(K, dtype = torch.float32, device = mean.device) / K
            else:
                weights = torch.tensor(weights_cfg, dtype = torch.float32, device = mean.device)
                weights = weights / weights.sum()

            k = torch.multinomial(weights, 1).item()

            if torch.is_tensor(std):
                s = float(std[k])
            elif isinstance(std, (list, tuple)):
                s = float(std[k])
            else:
                s = float(std)

            eps = torch.randn(self.dim, dtype = mean.dtype, device = mean.device) * s
            return mean[k] + eps

        elif dist_key == "checkerboard":
            if self._checkerboard is None:
                raise RuntimeError("CheckerboardDataset not initialized")
            return self._sample_checkerboard_tangent()

        else:
            raise ValueError(f"Unknown tangent-space distribution: {dist_name}")

    def sample_normalized(self, dist_name, std = None, mean = None, radius = None):
        z = self.sample_tangent(dist_name, std = std, mean = mean, radius = radius)
        z = self.normalize_tangent(z)
        x = self.tangent_to_manifold(z)
        return x            

    def __len__(self):
        if self.x0_all is not None:
            return int(self.x0_all.shape[0])
        return int(self.n_samples)

    def __getitem__(self, idx):
        if self.x0_all is None or self.x1_all is None:
            x0 = self.sample_normalized(self.x0_dist, std = self.std_x0, mean = self.mean_x0, radius = self.radius_x0)
            x1 = self.sample_normalized(self.x1_dist, std = self.std_x1, mean = self.mean_x1, radius = self.radius_x1)
            return {"x0": x0, "x1": x1}

        x0 = self.x0_all[idx]
        x1 = self.x1_all[idx]
        return {"x0": x0, "x1": x1}

    def _load_or_create_fixed_dataset(self):

        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        def _to_python(obj):
            if obj is None:
                return None
            if torch.is_tensor(obj):
                return obj.detach().cpu().tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (list, tuple)):
                return [_to_python(x) for x in obj]
            if isinstance(obj, (float, int, str, bool)):
                return obj
            # fallback for OmegaConf / other scalar-like objects
            try:
                return float(obj)
            except Exception:
                return str(obj)

        # build the exact metadata we expect for the current run.
        # this is used to decide whether an existing saved dataset is compatible.
        expected_meta = {
            "data": str(self.cfg.get("data", None)),
            "manifold": str(self.manifold_name),
            "curvature": float(self.gcfg.get("curvature", 1.0)),
            "dim": int(self.dim),
            "n_samples": int(self.n_samples),
            "eval_n_pairs_requested": int(self.eval_n_pairs),
            "x0_dist": str(self.x0_dist),
            "x1_dist": str(self.x1_dist),
            "std_x0": _to_python(self.std_x0),
            "std_x1": _to_python(self.std_x1),
            "mean_x0": _to_python(self.mean_x0),
            "mean_x1": _to_python(self.mean_x1),
            "radius_x0": _to_python(self.radius_x0),
            "radius_x1": _to_python(self.radius_x1),
            "origin": _to_python(self.reference_origin),
            "normalize_tangent_distributions": bool(
                self.gcfg.get("normalize_tangent_distributions", False)
            ),
            "eval_t_values": (
                None if self.eval_t_values is None else _to_python(self.eval_t_values)
            ),
            # use a dedicated dataset seed if provided; otherwise fall back to the main seed.
            "fixed_dataset_seed": int(self.cfg.get("eval_seed", self.cfg.get("seed", 0))),
        }

        meta_string = json.dumps(expected_meta, sort_keys = True, default = str)
        meta_hash = hashlib.md5(meta_string.encode("utf-8")).hexdigest()[:12]
        out_path = os.path.join(
            artifacts_dir,
            f"general_dataset_fixed_eval_{meta_hash}.pt",
        )

        # try to load an existing artifact, but only if metadata matches.
        if os.path.exists(out_path):
            payload = torch.load(out_path, map_location="cpu")
            saved_meta = payload.get("meta", {})

            # compare only the keys that define the sampled dataset.
            # if anything important changed, we regenerate.
            matches = all(saved_meta.get(k) == v for k, v in expected_meta.items())

            if matches:
                self.x0_all = payload.get("x0_all")
                self.x1_all = payload.get("x1_all")
                self.eval_x0 = payload.get("eval_x0")
                self.eval_x1 = payload.get("eval_x1")
                self.eval_t = payload.get("eval_t")

                # basic sanity checks in case the file exists but is incomplete/corrupt.
                if (
                    self.x0_all is not None
                    and self.x1_all is not None
                    and self.eval_x0 is not None
                    and self.eval_x1 is not None
                    and self.eval_t is not None
                ):
                    return

            # if we get here, the file exists but does not match the current config
            # (or is incomplete), so we regenerate below.

        # save RNG state so creating the fixed dataset does not disturb training RNG.
        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        py_state = random.getstate()

        fixed_seed = int(self.cfg.get("eval_seed", self.cfg.get("seed", 0)))
        torch.manual_seed(fixed_seed)
        np.random.seed(fixed_seed)
        random.seed(fixed_seed)

        try:
            # build the fixed evaluation time grid.
            # these are the probe times used later for x_t / u_t / v_theta evaluation.
            if self.eval_t_values is None:
                eval_t = torch.linspace(0.0, 1.0, 5, dtype=torch.float32)
            else:
                eval_t = torch.tensor(self.eval_t_values, dtype=torch.float32)

            # generate the full paired dataset once.
            # after this, __getitem__ will return these saved pairs instead of resampling.
            x0_all = []
            x1_all = []
            for _ in range(self.n_samples):
                x0 = self.sample_normalized(self.x0_dist, std = self.std_x0, mean = self.mean_x0, radius = self.radius_x0)
                x1 = self.sample_normalized(self.x1_dist, std = self.std_x1, mean = self.mean_x1, radius = self.radius_x1)

                x0_all.append(x0.detach().cpu())
                x1_all.append(x1.detach().cpu())

            x0_all = torch.stack(x0_all, dim=0).contiguous()
            x1_all = torch.stack(x1_all, dim=0).contiguous()

            # fixed evaluation subset:
            # use the first n_eval paired samples from the saved dataset.
            n_eval = min(int(self.eval_n_pairs), int(x0_all.shape[0]))
            eval_x0 = x0_all[:n_eval].contiguous()
            eval_x1 = x1_all[:n_eval].contiguous()

            # save full yaml config
            cfg_yaml = None
            try:
                from omegaconf import OmegaConf  # type: ignore

                cfg_yaml = OmegaConf.to_yaml(self.cfg)
            except Exception:
                cfg_yaml = None

            payload = {
                "meta": {
                    **expected_meta,
                    "eval_n_pairs_actual": int(n_eval),
                    "cfg_yaml": cfg_yaml,
                },
                # full fixed sampled dataset used by __getitem__
                "x0_all": x0_all,
                "x1_all": x1_all,
                # fixed evaluation subset used later for x_t / u_t / v_theta comparisons
                "eval_x0": eval_x0,
                "eval_x1": eval_x1,
                # fixed set of probe times for field evaluation
                "eval_t": eval_t,
            }

            torch.save(payload, out_path)

            # attach to the dataset object
            self.x0_all = x0_all
            self.x1_all = x1_all
            self.eval_x0 = eval_x0
            self.eval_x1 = eval_x1
            self.eval_t = eval_t
            
        finally:
            # restore RNG state so this helper does not affect the rest of the run.
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)
            random.setstate(py_state)


def _get_dataset(cfg):
    expand_factor = 1
    
    if cfg.data == "general_fm":
        dataset = GeneralDataset(cfg)
    else:
        raise ValueError(f"Unknown dataset option '{cfg.data}'")
    return dataset, expand_factor


class ExpandDataset(Dataset):
    def __init__(self, dataset: Dataset, expand_factor: int = 1):
        self.dataset = dataset
        self.expand_factor = max(1, int(expand_factor))

    def __len__(self):
        return len(self.dataset) * self.expand_factor

    def __getitem__(self, idx):
        base_len = len(self.dataset)
        if base_len == 0:
            raise IndexError("Empty dataset")
        return self.dataset[int(idx) % base_len]


def get_loaders(cfg):
    dataset, expand_factor = _get_dataset(cfg)

    N = len(dataset)
    N_val = N_test = N // 10
    N_train = N - N_val - N_test

    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    if data_seed is None:
        raise ValueError("seed for data generation must be provided")
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [N_train, N_val, N_test],
        generator=torch.Generator().manual_seed(data_seed),
    )

    # Expand the training set (we optimize based on number of iterations anyway).
    train_set = ExpandDataset(train_set, expand_factor=expand_factor)

    train_loader = DataLoader(
        train_set,
        cfg.optim.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=cfg.get("num_workers", 8),
    )
    val_loader = DataLoader(
        val_set,
        cfg.optim.val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.get("num_workers", 8),
    )
    test_loader = DataLoader(
        test_set,
        cfg.optim.val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.get("num_workers", 8),
    )

    return train_loader, val_loader, test_loader


def get_manifold(cfg):
    gcfg = cfg.general

    manifold_name = str(gcfg.manifold)
    curvature = float(gcfg.get("curvature", 1.0))
    dim = int(gcfg.dim)

    if manifold_name == "sphere":
        manifold = SphereCurvature(c = curvature)
    elif manifold_name == "poincare":
        manifold = PoincareBall(c = curvature)
    elif manifold_name == "euclidean":
        manifold = Euclidean()
    else:
        raise ValueError(f"unknown manifold: {manifold_name}")

    return manifold, dim