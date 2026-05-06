"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from csv import reader
import random
import numpy as np
import pandas as pd
import igl
import torch
import geoopt
from torch.utils.data import Dataset, DataLoader

from manifm.manifolds import Sphere, FlatTorus, Mesh, SPD, PoincareBall, Euclidean, SphereCurvature
from manifm.manifolds.mesh import Metric
from manifm.utils import cartesian_from_latlon
from manifm.manifolds.poincare import PoincareBallManifold





def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = np.array(list(lines)[1:]).astype(np.float64)
    return dataset


class MeshDataset(Dataset):
    dim = 3

    def __init__(self, root: str, data_file: str, obj_file: str, scale=1 / 250):
        with open(os.path.join(root, data_file), "rb") as f:
            data = np.load(f)

        v, f = igl.read_triangle_mesh(os.path.join(root, obj_file))

        self.v = torch.tensor(v).float() * scale
        self.f = torch.tensor(f).long()
        self.data = torch.tensor(data).float() * scale

    def manifold(self, *args, **kwargs):
        return Mesh(self.v, self.f, *args, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class HyperbolicDatasetPair(Dataset):
    manifold = PoincareBall()
    dim = 2
    #dim = 8

    def __init__(self, distance=0.6, std=0.7):
        self.distance = distance
        self.std = std

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        #sign0 = (torch.rand(1) > 0.5).float() * 2 - 1
        #sign1 = (torch.rand(1) > 0.5).float() * 2 - 1

        #mean0 = torch.tensor([self.distance, self.distance]) #* sign0
        mean0 = torch.tensor([0.0,0.0])
        #mean1 = torch.tensor([-self.distance, self.distance]) * sign1
        mean1 = torch.tensor([-self.distance, -self.distance])

        x0 = PoincareBall().wrapped_normal(2, mean=mean0, std=self.std)
        x1 = PoincareBall().wrapped_normal(2, mean=mean1, std=self.std)
        

        return {"x0": x0, "x1": x1}
    


#Old hyperbolic images class, before it was matching from uniform random noise to images
class HyperbolicImages(Dataset):
    dim = 512

    """
    Dataset for real hyperbolic embeddings on a Poincaré ball of curvature -c.
    """

    def __init__(self, emb_path, label_path=None, pair_mode="self"):
        """
        Args:
            emb_path: path to saved embeddings (torch.save)
            label_path: optional label file
            pair_mode:
                "self" → x0 is tangent noise, x1 is embedding
                "paired" → sample two different classes
                "none" → return only x1 
        """
        self.emb = torch.tensor(torch.load(emb_path)).float()

        # Ensure shapes [N, D]
        if self.emb.ndim == 1:
            self.emb = self.emb.unsqueeze(0)
        elif self.emb.ndim > 2:
            self.emb = self.emb.reshape(self.emb.shape[0], -1)

        self.labels = None
        if label_path is not None:
            self.labels = torch.tensor(torch.load(label_path))

        self.manifold = PoincareBall()
        self.dim = self.emb.shape[1]
        self.pair_mode = pair_mode

    def __len__(self):
        return len(self.emb)
    
    
    def __getitem__(self, idx, dim=512):

        x1 = self.emb[idx].reshape(-1)

        if self.pair_mode == "none":
            return {"x1": x1}

        if self.pair_mode == "self":
            # Uniform distribution from VRFM
            #x0 = 2*torch.rand(dim) - 1
            #x0 = PoincareBallManifold().wrap(x0)
            x0 = self.manifold.wrapped_normal(self.dim, mean=torch.zeros(self.dim), std=0.3)
            return {"x0": x0, "x1": x1}

        # Pair two different embeddings (e.g., across classes)
        if self.pair_mode == "paired":
            j = torch.randint(0, len(self.emb), (1,)).item()
            x0 = self.emb[j]
            return {"x0": x0, "x1": x1}


class EuclideanImages(Dataset):
    dim = 9216 #512x18

    """
    Dataset for real Euclidean embeddings.
    """

    def __init__(self, emb_path, label_path=None):
        """
        Args:
            emb_path: path to saved embeddings (torch.save)
            label_path: optional label file
            pair_mode:
                "self" → x0 is tangent noise, x1 is embedding
                "paired" → sample two different classes
                "none" → return only x1 
        """
        self.emb = torch.tensor(torch.load(emb_path)).float()

        # Ensure shapes [N,512]
        if self.emb.ndim == 1:
            self.emb = self.emb.unsqueeze(0)

        self.labels = None
        if label_path is not None:
            self.labels = torch.tensor(torch.load(label_path))

        self.manifold = Euclidean()
        self.dim = self.emb.shape[1]

    def __len__(self):
        return len(self.emb)
    
    
    def __getitem__(self, idx, dim=512):

        x1 = self.emb[idx]
        x0 = self.manifold.random_normal(self.dim, mean=torch.zeros(self.dim), std=1.0)
        return {"x0": x0, "x1": x1}
            

class HyperbolicUniformToGaussian(Dataset):
    """
    Synthetic dataset for learning a flow from
    Uniform(Poincaré Ball) → Wrapped Gaussian(Poincaré Ball)
    """
    
    def __init__(self, dim=2, mean=None, std=0.3, n_samples=20000):
        #super().__init__()
        self.dim = dim
        self.n_samples = n_samples
        self.std = std
        if mean is None:
            mean = torch.zeros(dim)
        self.mean = mean.float()
        self._manifold = PoincareBall()
    @property
    def manifold(self):
        return self._manifold

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        ### 1. Sample x0 ~ Uniform on ball - using VRFM implementation
        x0 = self._manifold.random_base(batch_size=1, dim=self.dim).squeeze(0)
        device = x0.device     
        mean = self.mean.to(device)
        ### 2. Sample x1 ~ Wrapped Normal
        x1 = self._manifold.wrapped_normal(self.dim, mean=mean, std=self.std)
        return {"x0": x0, "x1": x1}



class Wrapped(Dataset):
    def __init__(
        self,
        manifold,
        dim,
        n_mixtures=1,
        scale=0.2,
        centers=None,
        dataset_size=200000,
    ):
        self.manifold = manifold
        self.dim = dim
        self.n_mixtures = n_mixtures
        if centers is None:
            self.centers = self.manifold.random_uniform(n_mixtures, dim)
        else:
            self.centers = centers
        self.scale = scale
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        del idx

        idx = torch.randint(self.n_mixtures, (1,)).to(self.centers.device)
        mean = self.centers[idx].squeeze(0)

        tangent_vec = torch.randn(self.dim).to(self.centers)
        tangent_vec = self.manifold.proju(mean, tangent_vec)
        tangent_vec = self.scale * tangent_vec
        sample = self.manifold.expmap(mean, tangent_vec)
        return sample


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
            raise ValueError("Unknown manifold")

        if self.save_artifacts:
            self._load_or_create_fixed_dataset()
        else:
            self.x0_all = None
            self.x1_all = None
            self.eval_x0 = None
            self.eval_x1 = None
            self.eval_t = None
    

    def check_mean(self, mean, manifold, tol=1e-5):
        """
        Validates that mean(s) lie on the correct manifold.

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
                raise ValueError(
                    f"Poincaré mean(s) must have norm < 1. Got norms: {norms}"
                )

        elif manifold == "sphere":
            if not torch.all(torch.abs(norms - np.sqrt(1/self.curvature)) < tol):
                raise ValueError(
                    f"Sphere mean(s) must have norm ≈ {np.sqrt(1/self.curvature)}. Got norms: {norms}"
                )

        elif manifold == "euclidean":
            pass

        else:
            raise ValueError(f"Unknown manifold: {manifold}")
            

    def sample(self, dist_name, std=None, mean=None):
        if std is None:
            std = 1.0

        if mean is not None and not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=torch.float32)

        self.check_mean(mean, self.manifold_name)
        # --- UNIFORM ---
        if dist_name == "uniform":
            raise NotImplementedError("Uniform sampling not implemented yet")

        # --- EUCLIDEAN NORMAL ---
        elif dist_name == "normal":
            raise NotImplementedError("Euclidean normal not implemented yet")

        # --- RIEMANNIAN GAUSSIAN ---
        elif dist_name == "gaussian":
            if self.manifold_name == "euclidean":
                sample = self.manifold.random_normal(self.dim, mean=mean, std=std)
            else:
                sample = self.manifold.wrapped_normal(self.dim, mean=mean, std=std)
            return sample
        
        # --- MIXTURE OF GAUSSIANS ---
        # Define stds and means as list of lists. For example:
        # std = [[0.1, 0.1], [0.2, 0.2]]
        # mean = [[0.1, 0.1], [0.9, 0.9]]

        elif dist_name == "MoG":
            K = len(std)

            weights_cfg = self.gcfg.get("weights", None)
            if weights_cfg is None:
                weights = torch.ones(K) / K
            else:
                weights = torch.tensor(weights_cfg)
                weights = weights / weights.sum()

            # Sample one component
            k = torch.multinomial(weights, 1).item()

            m = mean[k]
            s = std[k]

            if self.manifold_name == "euclidean":
                sample = self.manifold.random_normal(self.dim, mean=m, std=s)
            else:
                sample = self.manifold.wrapped_normal(self.dim, mean=m, std=s)

            return sample



        else:
            raise ValueError(f"Unknown distribution: {dist_name}")

    def __len__(self):
        if self.x0_all is not None:
            return int(self.x0_all.shape[0])
        return int(self.n_samples)

    def __getitem__(self, idx):
        if self.x0_all is None or self.x1_all is None:
            x0 = self.sample(self.x0_dist, std=self.std_x0, mean=self.mean_x0)
            x1 = self.sample(self.x1_dist, std=self.std_x1, mean=self.mean_x1)
            return {"x0": x0, "x1": x1}

        x0 = self.x0_all[idx]
        x1 = self.x1_all[idx]
        return {"x0": x0, "x1": x1}

    def _load_or_create_fixed_dataset(self):

        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        out_path = os.path.join(artifacts_dir, "general_dataset_fixed_eval.pt")

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
            "eval_t_values": (
                None if self.eval_t_values is None else _to_python(self.eval_t_values)
            ),
            # use a dedicated dataset seed if provided; otherwise fall back to the main seed.
            "fixed_dataset_seed": int(self.cfg.get("eval_seed", self.cfg.get("seed", 0))),
        }

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
                x0 = self.sample(self.x0_dist, std=self.std_x0, mean=self.mean_x0)
                x1 = self.sample(self.x1_dist, std=self.std_x1, mean=self.mean_x1)

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
    if cfg.data == "volcano":
        dataset = Volcano(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 1550
    elif cfg.data == "earthquake":
        dataset = Earthquake(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 210
    elif cfg.data == "fire":
        dataset = Fire(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 100
    elif cfg.data == "flood":
        dataset = Flood(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 260
    elif cfg.data == "general":
        dataset = Top500(cfg.top500_datadir, amino="General")
        expand_factor = 1
    elif cfg.data == "glycine":
        dataset = Top500(cfg.top500_datadir, amino="Glycine")
        expand_factor = 10
    elif cfg.data == "proline":
        dataset = Top500(cfg.top500_datadir, amino="Proline")
        expand_factor = 18
    elif cfg.data == "prepro":
        dataset = Top500(cfg.top500_datadir, amino="Pre-Pro")
        expand_factor = 20
    elif cfg.data == "rna":
        dataset = RNA(cfg.rna_datadir)
        expand_factor = 14
    elif cfg.data == "simple_bunny":
        dataset = SimpleBunny(cfg.mesh_datadir)
    elif cfg.data == "bunny10":
        dataset = Bunny10(cfg.mesh_datadir)
    elif cfg.data == "bunny50":
        dataset = Bunny50(cfg.mesh_datadir)
    elif cfg.data == "bunny100":
        dataset = Bunny100(cfg.mesh_datadir)
    elif cfg.data == "spot10":
        dataset = Spot10(cfg.mesh_datadir)
    elif cfg.data == "spot50":
        dataset = Spot50(cfg.mesh_datadir)
    elif cfg.data == "spot100":
        dataset = Spot100(cfg.mesh_datadir)
    elif cfg.data == "maze3v2":
        dataset = Maze3v2(cfg.mesh_datadir)
    elif cfg.data == "maze4v2":
        dataset = Maze4v2(cfg.mesh_datadir)
    elif cfg.data == "wrapped_torus":
        manifold = FlatTorus()
        dataset = Wrapped(
            manifold,
            cfg.wrapped.dim,
            cfg.wrapped.n_mixtures,
            cfg.wrapped.scale,
            dataset_size=200000,
        )
    elif cfg.data == "wrapped_spd":
        manifold = SPD()
        d = cfg.wrapped.dim
        n = manifold.matdim(d)
        manifold = SPD(scale_std=0.5, scale_Id=3.0, base_expmap=False)
        centers = manifold.vectorize(torch.eye(n) * 2.0).reshape(1, -1)

        dataset = Wrapped(
            manifold,
            cfg.wrapped.dim,
            cfg.wrapped.n_mixtures,
            cfg.wrapped.scale,
            centers=centers,
            dataset_size=10000,
        )
    elif cfg.data == "eeg_1":
        dataset = EEG(cfg.eeg_datadir, set="1", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    elif cfg.data == "eeg_2a":
        dataset = EEG(cfg.eeg_datadir, set="2a", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    elif cfg.data == "eeg_2b":
        dataset = EEG(cfg.eeg_datadir, set="2b", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    elif cfg.data == "hyperbolic":
        dataset = HyperbolicDatasetPair()
    elif cfg.data == "images":
        dataset = HyperbolicImages(cfg.get("images_datadir"), cfg.get("images_labels"))
        #dataset = HyperbolicImages(cfg.get("images_datadir_A"), cfg.get("images_datadir_B"))
    elif cfg.data == "hyper_uni2norm":
        dataset = HyperbolicUniformToGaussian()
    elif cfg.data == "euclidean":
        dataset = EuclideanImages(cfg.get("euclidean_datadir"))
    elif cfg.data == "general_fm":
        dataset = GeneralDataset(cfg)

    else:
        raise ValueError("Unknown dataset option '{name}'")
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
        num_workers=cfg.get("num_workers", 8)
    )
    val_loader = DataLoader(
        val_set, 
        cfg.optim.val_batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.get("num_workers", 8)
    )
    test_loader = DataLoader(
        test_set,
        cfg.optim.val_batch_size, 
        shuffle=False, 
        pin_memory=True,
        num_workers=cfg.get("num_workers", 8)
    )

    return train_loader, val_loader, test_loader


def get_manifold(cfg):
    dataset, _ = _get_dataset(cfg)

    if isinstance(dataset, MeshDataset) or isinstance(dataset, MeshDatasetPair):
        manifold = dataset.manifold(
            numeigs=cfg.mesh.numeigs, metric=Metric(cfg.mesh.metric), temp=cfg.mesh.temp
        )
        return manifold, dataset.dim
    else:
        return dataset.manifold, dataset.dim
