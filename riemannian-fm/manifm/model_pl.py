"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, List
import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Point

from tqdm import tqdm
import geoopt
import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MinMetric
import pytorch_lightning as pl
from torch.func import vjp, jvp, vmap, jacrev
from torchdiffeq import odeint
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

from manifm.datasets import get_manifold
from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch
from manifm.utils import lonlat_from_cartesian, cartesian_from_latlon
from manifm.manifolds import (
    Sphere,
    FlatTorus,
    Euclidean,
    ProductManifold,
    Mesh,
    SPD,
    PoincareBall,
)
from manifm.manifolds.spd import plot_cone
from manifm.manifolds import geodesic
from manifm.mesh_utils import trimesh_to_vtk, points_to_vtk
from manifm.solvers import projx_integrator_return_last, projx_integrator

def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def output_and_div(vecfield, x, v=None, div_mode="exact"):
    if div_mode == "exact":
        dx = vecfield(x)
        div = vmap(div_fn(vecfield))(x)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class ManifoldFMLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.manifold, self.dim = get_manifold(cfg)

        # Model of the vector field.
        self.model = EMA(
            Unbatch(  # Ensures vmap works.
                ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                    tMLP(  # Vector field in the ambient space.
                        self.dim,
                        d_model=cfg.model.d_model,
                        num_layers=cfg.model.num_layers,
                        actfn=cfg.model.actfn,
                        fourier=cfg.model.get("fourier", None),
                    ),
                    manifold=self.manifold,
                    metric_normalize=self.cfg.model.get("metric_normalize", False),
                )
            ),
            cfg.optim.ema_decay,
        )

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metric = MeanMetric()
        self.val_metric = MeanMetric()
        self.test_metric = MeanMetric()

        # for logging best so far validation accuracy
        self.val_metric_best = MinMetric()

        # paths for saved evaluation artifacts (relative to Hydra run dir/cwd).
        self._fixed_eval_path = os.path.join("artifacts", "general_dataset_fixed_eval.pt")
        self._final_eval_path = os.path.join("artifacts", "final_fixed_eval_outputs.pt")

    @property
    def vecfield(self):
        return self.model

    @property
    def device(self):
        return self.model.parameters().__next__().device

    # helper function to get the current wandb run, if it exists
    def _wandb_run(self):
        if wandb is None or getattr(self, "trainer", None) is None:
            return None
        for lg in getattr(self.trainer, "loggers", []) or []:
            if lg.__class__.__name__ == "WandbLogger":
                return lg.experiment
        return None

    # helper function for logging tensors to wandb
    def _wandb_log_pt(self, name: str, payload: dict):
        run = self._wandb_run()
        if run is None:
            return
        os.makedirs("debug", exist_ok=True)
        path = os.path.join("debug", f"{name}-step{self.global_step:07d}.pt")
        torch.save(payload, path)
        art = wandb.Artifact(name=f"{name}-step{self.global_step:07d}", type="debug")
        art.add_file(path)
        run.log_artifact(art)

    def _wandb_log_file(self, name: str, path: str, type_name: str = "data"):
        run = self._wandb_run()
        if run is None or wandb is None:
            return
        art = wandb.Artifact(name=name, type=type_name)
        art.add_file(path)
        run.log_artifact(art)

    def on_fit_start(self) -> None:
        # save-once items at the beginning are produced by GeneralDataset and written to
        # artifacts/general_dataset_fixed_eval.pt. If present, upload it as a W&B artifact.
        if getattr(self, "trainer", None) is None or not self.trainer.is_global_zero:
            return
        if os.path.exists(self._fixed_eval_path):
            self._wandb_log_file("fixed_eval_inputs", self._fixed_eval_path, type_name="fixed_eval")

    @torch.no_grad()
    def on_fit_end(self) -> None:
        # only run once on the main process in distributed training.
        if getattr(self, "trainer", None) is None or not self.trainer.is_global_zero:
            return

        # if there is no fixed-eval input file, there is nothing to compute.
        if not os.path.exists(self._fixed_eval_path):
            return

        os.makedirs(os.path.dirname(self._final_eval_path), exist_ok=True)

        # load the fixed evaluation inputs saved by GeneralDataset.
        fixed = torch.load(self._fixed_eval_path, map_location="cpu")
        eval_x0 = fixed.get("eval_x0")
        eval_x1 = fixed.get("eval_x1")
        eval_t = fixed.get("eval_t")

        # basic sanity check.
        if eval_x0 is None or eval_x1 is None or eval_t is None:
            return

        # move fixed eval data to the model device.
        eval_x0 = eval_x0.to(self.device)
        eval_x1 = eval_x1.to(self.device)
        eval_t = eval_t.to(self.device)

        # this fixed-eval saving path is currently implemented for simple manifolds.
        # mesh uses solve_path settings and would need a separate implementation.
        if isinstance(self.manifold, Mesh):
            return

        # we will collect one tensor per saved time value, then stack over time.
        xt_list = []
        ut_list = []
        vtheta_list = []
        t_list = []

        # flatten eval_t in case it is stored as [T] or [T, 1].
        eval_t_flat = eval_t.flatten()

        for t_scalar in eval_t_flat.tolist():
            t_scalar = float(t_scalar)

            # build a batch-sized time tensor for this single probe time.
            if isinstance(self.manifold, SPD):
                # SPD geodesic API expects shape [N]
                t_batch = torch.full(
                    (eval_x0.shape[0],),
                    t_scalar,
                    device=self.device,
                    dtype=eval_x0.dtype,
                )

                # x_t = geodesic(x0, x1; t)
                # u_t = d/dt geodesic(x0, x1; t)
                def spd_geodesic(t_in):
                    return self.manifold.geodesic(eval_x0, eval_x1, t_in)

                x_t, u_t = jvp(
                    spd_geodesic,
                    (t_batch,),
                    (torch.ones_like(t_batch),),
                )

            else:
                # for sphere / poincare / euclidean, we use the geodesic path
                # x_t = expmap(x0, t * logmap(x0, x1))
                # u_t = d/dt x_t
                t_batch = torch.full(
                    (eval_x0.shape[0], 1),
                    t_scalar,
                    device=self.device,
                    dtype=eval_x0.dtype,
                )

                shooting_tangent_vec = self.manifold.logmap(eval_x0, eval_x1)

                def path(t_in):
                    return self.manifold.expmap(eval_x0, t_in * shooting_tangent_vec)

                x_t, u_t = jvp(
                    path,
                    (t_batch,),
                    (torch.ones_like(t_batch),),
                )

            # ensure shape [N, D]
            x_t = x_t.reshape(eval_x0.shape[0], self.dim)
            u_t = u_t.reshape(eval_x0.shape[0], self.dim)

            # evaluate the final learned vector field on the same (t, x_t) tuples.
            vtheta = self.vecfield(t_batch, x_t).reshape(eval_x0.shape[0], self.dim)

            xt_list.append(x_t.detach().cpu())
            ut_list.append(u_t.detach().cpu())
            vtheta_list.append(vtheta.detach().cpu())
            t_list.append(torch.tensor(t_scalar, dtype=torch.float32))

        # stack over time dimension.
        x_t_all = torch.stack(xt_list, dim=0)        # [T, N, D]
        u_t_all = torch.stack(ut_list, dim=0)        # [T, N, D]
        vtheta_all = torch.stack(vtheta_list, dim=0) # [T, N, D]
        eval_t_out = torch.stack(t_list, dim=0)      # [T]

        # this produces one final sample per saved start point.
        x1_hat = self.sample(
            eval_x0.shape[0],
            device=self.device,
            x0=eval_x0,
        ).detach().cpu()

        payload = {
            "meta": {
                "global_step": int(getattr(self, "global_step", 0)),
                "eval_n_pairs": int(eval_x0.shape[0]),
                "n_t": int(eval_t_out.shape[0]),
                "dim": int(self.dim),
                "manifold_type": self.manifold.__class__.__name__,
            },
            # fixed evaluation inputs
            "eval_x0": fixed.get("eval_x0"),   
            "eval_x1": fixed.get("eval_x1"),
            "eval_t": eval_t_out,           
            # final evaluation outputs
            "x_t": x_t_all,
            "u_t": u_t_all,
            "vtheta": vtheta_all,
            "x1_hat": x1_hat,
        }

        torch.save(payload, self._final_eval_path)

        # upload to W&B
        self._wandb_log_file(
            "final_fixed_eval_outputs",
            self._final_eval_path,
            type_name="fixed_eval",
        )
    
    @torch.no_grad()
    def visualize(self, batch, force=False):
        if not force and not self.cfg.get("visualize", False):
            return

        if isinstance(self.manifold, Sphere) and self.dim == 3:
            self.plot_earth2d(batch)

        if isinstance(self.manifold, FlatTorus) and self.dim == 2:
            self.plot_torus2d(batch)

        if isinstance(self.manifold, Mesh) and self.dim == 3:
            self.plot_mesh(batch)

        if isinstance(self.manifold, SPD) and self.dim >= 3:
            self.plot_spd(batch)

        if isinstance(self.manifold, PoincareBall) and self.dim == 2:
            self.plot_poincare(batch)

    @torch.no_grad()
    def plot_poincare(self, batch):
        os.makedirs("figs", exist_ok=True)

        x0 = batch["x0"]
        x1 = batch["x1"]

        trajs = self.sample_all(x1.shape[0], device=x1.device, x0=x0)
        samples = trajs[-1]

        # Plot model samples
        x0 = x0.detach().cpu().numpy()
        x1 = x1.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        trajs = trajs.detach().cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], s=2, color="C3")
        plt.gca().add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"figs/samples-{self.global_step:06d}.png")
        plt.savefig(f"figs/samples-{self.global_step:06d}.pdf")
        plt.close()

        # Plot trajectories
        plt.figure(figsize=(6, 6))
        plt.scatter(x0[:, 0], x0[:, 1], s=2, color="C0")
        plt.scatter(x1[:, 0], x1[:, 1], s=2, color="C1")
        for i in range(100):
            plt.plot(trajs[:, i, 0], trajs[:, i, 1], color="grey", linewidth=0.5)
        plt.gca().add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"figs/trajs-{self.global_step:06d}.png")
        plt.savefig(f"figs/trajs-{self.global_step:06d}.pdf")
        plt.close()

    @torch.no_grad()
    def plot_sphere_3d(self, batch):
        os.makedirs("figs", exist_ok=True)

        x0 = batch["x0"]
        x1 = batch["x1"]

        trajs = self.sample_all(x1.shape[0], device=x1.device, x0=x0)
        samples = trajs[-1]

        x0 = x0.detach().cpu().numpy()
        x1 = x1.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        trajs = trajs.detach().cpu().numpy()

        c = getattr(self.manifold, "c", 1.0)
        if torch.is_tensor(c):
            c = c.item()
        R = 1.0 / np.sqrt(c) if c > 0 else 1.0

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        u, v = np.meshgrid(u, v)
        
        xs = R * np.cos(u) * np.sin(v)
        ys = R * np.sin(u) * np.sin(v)
        zs = R * np.cos(v)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_wireframe(xs, ys, zs, color="lightblue", alpha=0.1, linewidth=0.5)
        
        ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=12, color="red", label="Start (x0)")
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=12, color="blue", label="End (x1)")
        
        for i in range(min(50, trajs.shape[1])):
            ax.plot(trajs[:, i, 0], trajs[:, i, 1], trajs[:, i, 2], color="grey", alpha=0.5, linewidth=1)

        ax.set_xlim([-R, R])
        ax.set_ylim([-R, R])
        ax.set_zlim([-R, R])
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        
        plt.savefig(f"figs/sphere-{self.global_step:06d}.png")
        plt.close()

    def add_geodesic_grid(self,ax: plt.Axes, manifold: geoopt.Stereographic, line_width=0.1):

        # define geodesic grid parameters
        N_EVALS_PER_GEODESIC = 10000
        STYLE = "--"
        COLOR = "gray"
        LINE_WIDTH = line_width

        # get manifold properties
        print("curvature:", manifold)
        K = manifold.c
        R = manifold.radius

        # get maximal numerical distance to origin on manifold
        if K < 0:
            # create point on R
            r = torch.tensor((R, 0.0), dtype=manifold.dtype)
            # project point on R into valid range (epsilon border)
            r = manifold.projx(r)
            # determine distance from origin
            max_dist_0 = manifold.dist0(r).item()
        else:
            max_dist_0 = np.pi * R
        # adjust line interval for spherical geometry
        circumference = 2*np.pi*R

        # determine reasonable number of geodesics
        # choose the grid interval size always as if we'd be in spherical
        # geometry, such that the grid interpolates smoothly and evenly
        # divides the sphere circumference
        n_geodesics_per_circumference = 4 * 6  # multiple of 4!
        n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
        grid_interval_size = circumference / n_geodesics_per_circumference
        if K < 0:
            n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)

        # create time evaluation array for geodesics
        if K < 0:
            min_t = -1.2*max_dist_0
        else:
            min_t = -circumference/2.0
        t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]

        # define a function to plot the geodesics
        def plot_geodesic(gv):
            ax.plot(*gv.t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH)

        # define geodesic directions
        u_x = torch.tensor((0.0, 1.0))
        u_y = torch.tensor((1.0, 0.0))

        # add origin x/y-crosshair
        o = torch.tensor((0.0, 0.0))
        if K < 0:
            x_geodesic = manifold.geodesic_unit(t, o, u_x)
            y_geodesic = manifold.geodesic_unit(t, o, u_y)
            plot_geodesic(x_geodesic)
            plot_geodesic(y_geodesic)
        else:
            # add the crosshair manually for the sproj of sphere
            # because the lines tend to get thicker if plotted
            # as done for K<0
            ax.axvline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)
            ax.axhline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)

        # add geodesics per quadrant
        for i in range(1, n_geodesics_per_quadrant):
            i = torch.as_tensor(float(i))
            # determine start of geodesic on x/y-crosshair
            x = manifold.geodesic_unit(i*grid_interval_size, o, u_y)
            y = manifold.geodesic_unit(i*grid_interval_size, o, u_x)

            # compute point on geodesics
            x_geodesic = manifold.geodesic_unit(t, x, u_x)
            y_geodesic = manifold.geodesic_unit(t, y, u_y)

            # plot geodesics
            plot_geodesic(x_geodesic)
            plot_geodesic(y_geodesic)
            if K < 0:
                plot_geodesic(-x_geodesic)
                plot_geodesic(-y_geodesic)
    @torch.no_grad()
    def plot_sphere_2d(self, batch):
        os.makedirs("figs", exist_ok=True)

        x0 = batch["x0"]
        x1 = batch["x1"]
        trajs = self.sample_all(x1.shape, device=x1.device, x0=x0)
        samples = trajs[-1]

        c = getattr(self.manifold, "c", 1.0)
        if torch.is_tensor(c):
            c = c.item()
        R = 1.0 / np.sqrt(c) if c > 0 else 1.0

        def project(tensor):
            x, y, z = tensor[..., 0], tensor[..., 1], tensor[..., 2]
            denom = R - z + 1e-7
            return torch.stack([R * x / denom, R * y / denom], dim=-1)

        x0_2d = project(x0).cpu().numpy()
        samples_2d = project(samples).cpu().numpy()
        trajs_2d = project(trajs).cpu().numpy()

        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        equator = plt.Circle((0, 0), R, color="black", fill=False, linestyle='--', alpha=0.5)
        ax.add_artist(equator)

        self.add_geodesic_grid(ax, self.manifold, line_width=0.2)

        for i in range(min(100, trajs_2d.shape[1])):
            ax.plot(trajs_2d[:, i, 0], trajs_2d[:, i, 1], color="grey", alpha=0.3, linewidth=0.5)

        plt.scatter(x0_2d[:, 0], x0_2d[:, 1], s=5, color="red")
        plt.scatter(samples_2d[:, 0], samples_2d[:, 1], s=5, color="blue")

        plt.xlim([-3 * R, 3 * R])
        plt.ylim([-3 * R, 3 * R])
        ax.set_aspect('equal')
        plt.axis("off")
        plt.savefig(f"figs/sphere2d-{self.global_step:06d}.png")
        plt.close()

    @torch.no_grad()
    def plot_spd(self, batch):
        os.makedirs("figs", exist_ok=True)

        ax = plot_cone()

        samples = self.sample(batch.shape[0], device=batch.device)

        # Take a 2x2 slice
        samples = self.manifold.devectorize(samples)[..., :2, :2]
        samples = self.manifold.vectorize(samples)

        samples = samples.cpu().numpy()
        c = samples[:, 1]
        u = 0.5 * (samples[:, 0] + samples[:, 2])
        v = 0.5 * (samples[:, 0] - samples[:, 2])
        ax.scatter(c, v, u, marker=".", c="C1", s=3)

        # Take a 2x2 slice
        batch = self.manifold.devectorize(batch)[..., :2, :2]
        batch = self.manifold.vectorize(batch)

        batch = batch.cpu().numpy()
        c = batch[:, 1]
        u = 0.5 * (batch[:, 0] + batch[:, 2])
        v = 0.5 * (batch[:, 0] - batch[:, 2])
        ax.scatter(c, v, u, marker=".", c="k", s=3)

        plt.tight_layout()
        plt.savefig(f"figs/samples-{self.global_step:06d}.png")
        plt.close()

    @torch.no_grad()
    def plot_mesh(self, batch):
        os.makedirs("figs", exist_ok=True)

        if isinstance(batch, dict):
            noise = batch["x0"]
            data = batch["x1"]
        else:
            noise = None
            data = batch

        # Generate model samples
        trajs = self.sample_all(data.shape[0], data.device, x0=noise)
        os.makedirs("figs/trajs", exist_ok=True)
        for i in range(trajs.shape[0]):
            xt = trajs[i]
            points_to_vtk(f"figs/trajs/{self.cfg.data}-samples-{i:04d}", xt)

        # samples = self.sample(data.shape[0], data.device, x0=noise)
        # points_to_vtk(f"figs/{self.cfg.data}-samples", samples)

        # Compute log probability at vertices
        v, f = self.manifold.v, self.manifold.f

        logprobs = []
        for x in tqdm(torch.split(v, 10000)):
            logprobs.append(self.compute_exact_loglikelihood(x))
        logprobs = torch.cat(logprobs, dim=0)
        probs = torch.exp(logprobs)
        point_data = {"logprobs": logprobs, "probs": probs}
        trimesh_to_vtk(f"figs/{self.cfg.data}-density", v, f, point_data=point_data)

        plt.tight_layout()
        plt.savefig(f"figs/samples-{self.global_step:06d}.png")
        plt.savefig(f"figs/samples-{self.global_step:06d}.pdf")
        plt.close()

    @torch.no_grad()
    def plot_earth2d(self, batch):
        os.makedirs("figs", exist_ok=True)

        # Plot world map
        world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
        ax = world.plot(figsize=(9, 4), antialiased=False, color="grey")

        # Plot model samples
        # samples = self.sample(batch.shape[0], batch.device)
        # samples = samples.cpu()
        # geometry = [Point(lonlat_from_cartesian(x) / np.pi * 180) for x in samples]
        # pts = geopandas.GeoDataFrame(geometry=geometry)
        # pts.plot(ax=ax, color="#1a9850", markersize=0.01, alpha=0.7)

        # Plot model likelihood
        N = 400
        x = np.linspace(-180.0, 180.0, N)  # longitude
        y = np.linspace(-90.0, 90.0, N)  # latitude
        X, Y = np.meshgrid(x, y)

        if os.path.exists(f"figs/{self.cfg.data}-logps-{N}-{self.global_step:06d}.npy"):
            L = np.load(f"figs/{self.cfg.data}-logps-{N}-{self.global_step:06d}.npy")
        else:
            lonlat = np.stack([Y.reshape(-1), X.reshape(-1)], axis=-1)
            xyz = cartesian_from_latlon(torch.tensor(lonlat) * np.pi / 180)
            logps = []
            for c in tqdm(torch.split(xyz, 8000)):
                c = c.to(batch)
                logps.append(self.compute_exact_loglikelihood(c).cpu().numpy())
            logps = np.concatenate(logps, axis=0)
            L = logps.reshape(N, N)
            np.save(f"figs/{self.cfg.data}-logps-{N}-{self.global_step:06d}.npy", L)

        P = np.exp(L)
        cs = ax.contourf(
            X,
            Y,
            P,
            levels=np.linspace(0, 1, 11),
            alpha=0.7,
            extend="max",
            cmap="BuGn",
            antialiased=True,
        )

        # Plot data samples
        batch = batch.cpu()
        geometry = [Point(lonlat_from_cartesian(x) / np.pi * 180) for x in batch]
        pts = geopandas.GeoDataFrame(geometry=geometry)
        pts.plot(ax=ax, color="#d73027", markersize=0.01, alpha=0.7)

        cbar = plt.colorbar(cs, ax=ax, pad=0.01, ticks=[0, 1])
        cbar.ax.set_yticklabels(["0", "$\geq$1"])
        cbar.ax.set_ylabel("likelihood", fontsize=18, rotation=270, labelpad=10)
        ax.tick_params(axis="both", which="both", direction="in", length=3)
        cbar.ax.tick_params(axis="both", which="both", direction="in", length=3)
        cbar.set_alpha(0.7)
        cbar.draw_all()

        # plt.axis("off")
        plt.xlim([-180, 180])
        plt.ylim([-90, 90])
        plt.xlabel("Longitude", fontsize=18)
        plt.ylabel("Latitude", fontsize=18)
        plt.tight_layout()
        plt.savefig(f"figs/{self.cfg.data}-samples-{self.global_step:06d}.png", dpi=300)
        plt.savefig(f"figs/{self.cfg.data}-samples-{self.global_step:06d}.pdf")
        plt.close()

    @torch.no_grad()
    def plot_torus2d(self, batch):
        os.makedirs("figs", exist_ok=True)

        plt.rcParams["axes.autolimit_mode"] = "round_numbers"

        plt.figure(figsize=(6.1, 5))
        ax = plt.gca()

        # Plot model samples
        # samples = self.sample(batch.shape[0], batch.device)
        # samples = samples.cpu().numpy()
        # plt.scatter(samples[..., 0], samples[..., 1], marker=".", c="C0", s=1)

        # Plot density
        N = 400
        x = np.linspace(-np.pi, np.pi, N)  # longitude
        y = np.linspace(-np.pi, np.pi, N)  # latitude
        X, Y = np.meshgrid(x, y)

        if os.path.exists(f"figs/{self.cfg.data}-logps-{N}-{self.global_step:06d}.npy"):
            L = np.load(f"figs/{self.cfg.data}-logps-{N}-{self.global_step:06d}.npy")
        else:
            inputs = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
            # wrap to [0, 2pi]
            inputs = inputs % (2 * np.pi)
            inputs = torch.tensor(inputs).to(batch)
            logps = []
            for c in tqdm(torch.split(inputs, 8000)):
                logps.append(self.compute_exact_loglikelihood(c).cpu().numpy())
            logps = np.concatenate(logps, axis=0)
            L = logps.reshape(N, N)
            np.save(f"figs/{self.cfg.data}-logps-{N}-{self.global_step:06d}.npy", L)

        X = X / np.pi * 180
        Y = Y / np.pi * 180
        cs = ax.contourf(X, Y, L, alpha=0.9, cmap="Blues", antialiased=True)

        # Plot data samples
        batch = batch.cpu().numpy()[:10000]
        batch = (batch + np.pi) % (2 * np.pi) - np.pi
        batch = batch / np.pi * 180

        plt.scatter(
            batch[..., 0], batch[..., 1], marker=".", c="#d73027", s=0.05, alpha=0.7
        )
        plt.xlim([-180, 180])
        plt.ylim([-180, 180])
        ax.set_aspect("equal")
        plt.xlabel(r"$\phi$", fontsize=18)
        plt.ylabel(r"$\psi$", fontsize=18, rotation=0)

        plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.8, linewidth=0.5)
        plt.axvline(x=0.0, color="black", linestyle="--", alpha=0.8, linewidth=0.5)

        cbar = plt.colorbar(cs, ax=ax, pad=0.01)
        cbar.ax.set_ylabel("log likelihood", fontsize=18, rotation=270, labelpad=10)
        ax.tick_params(axis="both", which="both", direction="in", length=3)
        cbar.ax.tick_params(axis="both", which="both", direction="in", length=3)

        plt.tight_layout()
        plt.savefig(f"figs/{self.cfg.data}-{self.global_step:06d}.png", dpi=300)
        plt.savefig(f"figs/{self.cfg.data}-{self.global_step:06d}.pdf")
        plt.close()

    @torch.no_grad()
    def compute_cost(self, batch):
        if isinstance(batch, dict):
            x0 = batch["x0"]
        else:
            x0 = (
                self.manifold.random_base(batch.shape[0], self.dim)
                .reshape(batch.shape[0], self.dim)
                .to(batch.device)
            )

        # Solve ODE.
        x1 = odeint(
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, 2).to(x0.device),
            atol=self.cfg.model.atol,
            rtol=self.cfg.model.rtol,
        )[-1]

        x1 = self.manifold.projx(x1)

        return self.manifold.dist(x0, x1)

    @torch.no_grad()
    def sample(self, n_samples, device, x0=None):
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.manifold.random_base(n_samples, self.dim)
                .reshape(n_samples, self.dim)
                .to(device)
            )

        local_coords = self.cfg.get("local_coords", False)
        eval_projx = self.cfg.get("eval_projx", False)

        # Solve ODE.
        if not eval_projx and not local_coords:
            # If no projection, use adaptive step solver.
            x1 = odeint(
                self.vecfield,
                x0,
                t=torch.linspace(0, 1, 2).to(device),
                atol=self.cfg.model.atol,
                rtol=self.cfg.model.rtol,
                options={"min_step": 1e-5}
            )[-1]
        else:
            # If projection, use 1000 steps.
            x1 = projx_integrator_return_last(
                self.manifold,
                self.vecfield,
                x0,
                t=torch.linspace(0, 1, 1001).to(device),
                method="euler",
                projx=eval_projx,
                local_coords=local_coords,
                pbar=True,
            )

        # SCALING - was commented before
        x1 = self.manifold.projx(x1)
        return x1

    @torch.no_grad()
    def sample_all(self, n_samples, device, x0=None):
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.manifold.random_base(n_samples, self.dim)
                .reshape(n_samples, self.dim)
                .to(device)
            )

        # Solve ODE.
        xs, _ = projx_integrator(
            self.manifold,
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, 1001).to(device),
            method="euler",
            projx=True,
            pbar=True,
        )
        return xs

    @torch.no_grad()
    def compute_exact_loglikelihood(
        self,
        batch: torch.Tensor,
        t1: float = 1.0,
        return_projx_error: bool = False,
        num_steps=1000,
    ):
        """Computes the negative log-likelihood of a batch of data."""

        try:
            nfe = [0]

            div_mode = self.cfg.get("div_mode", "exact")

            with torch.inference_mode(mode=False):
                v = None
                if div_mode == "rademacher":
                    v = torch.randint(low=0, high=2, size=batch.shape).to(batch) * 2 - 1

                def odefunc(t, tensor):
                    nfe[0] += 1
                    t = t.to(tensor)
                    x = tensor[..., : self.dim]
                    vecfield = lambda x: self.vecfield(t, x)
                    dx, div = output_and_div(vecfield, x, v=v, div_mode=div_mode)

                    if hasattr(self.manifold, "logdetG"):

                        def _jvp(x, v):
                            return jvp(self.manifold.logdetG, (x,), (v,))[1]

                        corr = vmap(_jvp)(x, dx)
                        div = div + 0.5 * corr.to(div)

                    div = div.reshape(-1, 1)
                    del t, x
                    return torch.cat([dx, div], dim=-1)

                # Solve ODE on the product manifold of data manifold x euclidean.
                product_man = ProductManifold(
                    (self.manifold, self.dim), (Euclidean(), 1)
                )
                state1 = torch.cat([batch, torch.zeros_like(batch[..., :1])], dim=-1)

                local_coords = self.cfg.get("local_coords", False)
                eval_projx = self.cfg.get("eval_projx", False)

                with torch.no_grad():
                    if not eval_projx and not local_coords:
                        # If no projection, use adaptive step solver.
                        state0 = odeint(
                            odefunc,
                            state1,
                            t=torch.linspace(t1, 0, 2).to(batch),
                            atol=self.cfg.model.atol,
                            rtol=self.cfg.model.rtol,
                            method="dopri5",
                            options={"min_step": 1e-5},
                        )[-1]
                    else:
                        # If projection, use 1000 steps.
                        state0 = projx_integrator_return_last(
                            product_man,
                            odefunc,
                            state1,
                            t=torch.linspace(t1, 0, num_steps + 1).to(batch),
                            method="euler",
                            projx=eval_projx,
                            local_coords=local_coords,
                            pbar=True,
                        )

                # log number of function evaluations
                self.log("nfe", nfe[0], prog_bar=True, logger=True)

                x0, logdetjac = state0[..., : self.dim], state0[..., -1]
                x0_ = x0
                x0 = self.manifold.projx(x0)

                # log how close the final solution is to the manifold.
                integ_error = (x0[..., : self.dim] - x0_[..., : self.dim]).abs().max()
                self.log("integ_error", integ_error)

                logp0 = self.manifold.base_logprob(x0)
                logp1 = logp0 + logdetjac

                if self.cfg.get("normalize_loglik", False):
                    logp1 = logp1 / self.dim

                # Mask out those that left the manifold
                masked_logp1 = logp1
                if isinstance(self.manifold, SPD):
                    mask = integ_error < 1e-5
                    self.log("frac_within_manifold", mask.sum() / mask.nelement())
                    masked_logp1 = logp1[mask]

                if return_projx_error:
                    return logp1, integ_error
                else:
                    return masked_logp1
        except:
            traceback.print_exc()
            return torch.zeros(batch.shape[0]).to(batch)

    def loss_fn(self, batch: torch.Tensor):
        return self.rfm_loss_fn(batch)

    def rfm_loss_fn(self, batch: torch.Tensor):
        if isinstance(batch, dict):
            x0 = batch["x0"]
            x1 = batch["x1"]
        else:
            x1 = batch
            x0 = self.manifold.random_base(x1.shape[0], self.dim).to(x1)

        N = x1.shape[0]

        if isinstance(self.manifold, Mesh):
            t1 = 1.0 - self.cfg.mesh.time_eps
            T = self.cfg.mesh.nsteps
            # stratified sample of time values
            t = torch.linspace(0, t1, T + 1)
            t = t + torch.rand_like(t) * t1 / T
            t[-1] = t1
            t = F.pad(t, (1, 0), value=0.0)
            t = t.to(x1)
            T = t.shape[0]

            with torch.no_grad():
                x_t, u_t = self.manifold.solve_path(
                    x0,
                    x1,
                    t,
                    projx=self.cfg.mesh.projx,
                    method=self.cfg.mesh.method,
                    atol=self.cfg.mesh.atol,
                    rtol=self.cfg.mesh.rtol,
                )

            x_t = x_t.reshape(T * N, 3)
            u_t = u_t.reshape(T * N, 3)
            t = t.reshape(T, 1).expand(T, N).reshape(-1)

        elif isinstance(self.manifold, SPD):
            t = torch.rand(N).to(x1)

            def SPD_geodesic(t):
                return self.manifold.geodesic(x0, x1, t)

            x_t, u_t = jvp(SPD_geodesic, (t,), (torch.ones_like(t).to(t),))
            x_t = x_t.reshape(N, self.dim)
            u_t = u_t.reshape(N, self.dim)

        else:
            t = torch.rand(N).reshape(-1, 1).to(x1)

            # batched geodesic evaluation + derivative w.r.t. time.
            shooting_tangent_vec = self.manifold.logmap(x0, x1)

            def path(t):
                return self.manifold.expmap(x0, t * shooting_tangent_vec)

            x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            x_t = x_t.reshape(N, self.dim)
            u_t = u_t.reshape(N, self.dim)

            # t = torch.rand(N, 1, device=x1.device, dtype=x1.dtype)

            # x_t_list = []
            # u_t_list = []

            # for i in range(N):
            #     path = geodesic(self.manifold, x0[i], x1[i])

            #     ti = t[i]  # shape [1]
            #     xi, ui = jvp(path, (ti,), (torch.ones_like(ti),))

            #     x_t_list.append(xi)
            #     u_t_list.append(ui)

            # x_t = torch.cat(x_t_list, dim=0)  # [N, dim]
            # u_t = torch.cat(u_t_list, dim=0)

        diff = self.vecfield(t, x_t) - u_t
        return self.manifold.inner(x_t, diff, diff).mean() / self.dim

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.loss_fn(batch)

        if torch.isfinite(loss):
            # log train metrics
            self.log("train/loss", loss, on_step=True, on_epoch=True)
            self.train_metric.update(loss)
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss.item()}.")
            return None

        return {"loss": loss}

    # def training_epoch_end(self, outputs):
    def on_train_epoch_end(self):
        self.train_metric.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        if isinstance(batch, dict):
            x1 = batch["x1"]
        else:
            x1 = batch

        logprob = self.compute_exact_loglikelihood(x1)
        loss = -logprob.mean()
        batch_size = x1.shape[0]

        # log the RFM loss on validation batches
        with torch.no_grad():
            rfm_loss = self.rfm_loss_fn(batch)
        self.log(
            "val/rfm_loss",
            rfm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.val_metric.update(-logprob)

        if batch_idx == 0:
            self.visualize(batch)
            
            # log model samples 
            run = self._wandb_run()
            if run is not None:
                n = 256
                if isinstance(batch, dict) and "x0" in batch:
                    x0 = batch["x0"][:n]
                    x1_hat = self.sample(x0.shape[0], device=x0.device, x0=x0)
                    self._wandb_log_pt("val_x1_hat", {"x0": x0.detach().cpu(), "x1_hat": x1_hat.detach().cpu()})
                else:
                    x1_hat = self.sample(n, device=self.device)
                    self._wandb_log_pt("val_x1_hat", {"x1_hat": x1_hat.detach().cpu()})

            # log integration grid used for trajectories
            t_grid = torch.linspace(0, 1, 1001)
            self._wandb_log_pt("eval_t_grid", {"t_grid": t_grid})   
        return {"loss": loss}

    # def validation_epoch_end(self, outputs):
    def on_validation_epoch_end(self):
        val_loss = self.val_metric.compute()
        self.val_metric_best.update(val_loss)
        self.log("val/loss_best", self.val_metric_best.compute(), on_epoch=True, prog_bar=True)
        self.val_metric.reset()

    def test_step(self, batch: Any, batch_idx: int):
        if isinstance(batch, dict):
            batch = batch["x1"]

        logprob = self.compute_exact_loglikelihood(batch)
        loss = -logprob.mean()
        batch_size = batch.shape[0]

        self.log("test/loss", loss, batch_size=batch_size)
        self.test_metric.update(-logprob)
        return {"loss": loss}

    # def test_epoch_end(self):
    def on_test_epoch_end(self):
        self.test_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.wd,
            eps=self.cfg.optim.eps,
        )

        if self.cfg.optim.get("scheduler", "cosine") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.optim.num_iterations,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return {
                "optimizer": optimizer,
            }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.model, EMA):
            self.model.update_ema()
