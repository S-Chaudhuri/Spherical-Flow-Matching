"""Copyright (c) Meta Platforms, Inc. and affiliates."""
# train.py


import os

# Use PyTorch backend for geomstats
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import os.path as osp
import sys
import math
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import json
from glob import glob
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from manifm.datasets import get_loaders
from manifm.model_pl import ManifoldFMLitModule



torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    logging.getLogger("pytorch_lightning").setLevel(logging.getLevelName("INFO"))

    if cfg.get("seed", None) is not None:
        #pl.utilities.seed.seed_everything(cfg.seed)
        pl.seed_everything(cfg.seed)

    def configure_dynamically(cfg):
        dim = cfg.general.dim
        dist = 0.5

        if cfg.general.x1_dist == "gaussian-ring" or cfg.general.x1_dist == "gaussian":
            if cfg.general.manifold == "euclidean":
                cfg.general.mean_x0 = [0.0] * dim
                cfg.general.mean_x1 = [float(dist / math.sqrt(dim))] * dim
            
            if cfg.general.manifold == "poincare":
                # curvature = cfg.general.curvature

                # u = torch.ones(dim)
                # u = u / torch.norm(u) 

                # rho = torch.tanh(torch.sqrt(torch.tensor(curvature)) * dist / 2.0) / torch.sqrt(torch.tensor(curvature))
                # x1 = rho * u

                # cfg.general.mean_x0 = [0.0] * dim
                # cfg.general.mean_x1 = [float(f"{v:.8f}") for v in x1]
                cfg.general.mean_x0 = [0.0] * dim
                cfg.general.mean_x1 = [float(dist / math.sqrt(dim))] * dim


            if cfg.general.manifold == "sphere":
                # curvature = cfg.general.curvature

                # # sphere radius
                # R = 1.0 / math.sqrt(curvature)

                # # geodesic angle corresponding to distance dist
                # theta = dist / R

                # # north pole (mu0)
                # mu0 = torch.zeros(dim)
                # mu0[0] = R

                # # symmetric point at distance 'dist' from mu0
                # mu1 = torch.zeros(dim)
                # mu1[0] = R * math.cos(theta)

                # if dim < 2:
                #     raise ValueError("Sphere construction requires dim >= 2")

                # spread_val = R * math.sin(theta) / math.sqrt(dim - 1)
                # mu1[1:] = spread_val

                # # save to cfg
                # cfg.general.mean_x0 = [float(f"{v:.7f}") for v in mu0]
                # cfg.general.mean_x1 = [float(f"{v:.7f}") for v in mu1]

                # if cfg.general.get("radius_x1", None) is not None:
                #     pi_R = math.pi * R
                #     cfg.general.radius_x1 = float(
                #         pi_R * torch.tanh(torch.tensor(cfg.general.radius_x1) / pi_R)
                #     )
                # Unit direction in the subspace orthogonal to the first axis.
                direction = [0.0] * dim
                spread_val = 1.0 / math.sqrt(dim - 1)
                for i in range(1, dim):
                    direction[i] = spread_val

                cfg.general.mean_x0 = [0.0] * dim
                cfg.general.mean_x1 = [float(dist * v) for v in direction]

        else:
            pass  # use YAML

    configure_dynamically(cfg)

    print(cfg)

    keys = [
        "SLURM_NODELIST",
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_JOB_NAME",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
    ]
    log.info(json.dumps({k: os.environ.get(k, None) for k in keys}, indent=4))

    cmd_str = " \\\n".join([f"python {sys.argv[0]}"] + ["\t" + x for x in sys.argv[1:]])
    with open("cmd.sh", "w") as fout:
        print("#!/bin/bash\n", file=fout)
        print(cmd_str, file=fout)

    log.info(f"CWD: {os.getcwd()}")

    # Load dataset
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # Construct model
    model = ManifoldFMLitModule(cfg)
    print(model)

    # Checkpointing, logging, and other misc.
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            monitor="val/loss_best",
            mode="min",
            filename="epoch-{epoch:03d}_step-{global_step}_loss-{val_loss:.4f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            every_n_train_steps=cfg.get("ckpt_every", None),
        ),
        LearningRateMonitor(),
    ]

    # slurm_plugin = pl.plugins.environments.SLURMEnvironment(auto_requeue=False)
    _SLURMEnvironment = None
    try:
        from pytorch_lightning.plugins.environments import SLURMEnvironment as _SLURMEnvironment  # type: ignore
    except Exception:
        try:
            from lightning_fabric.plugins.environments import SLURMEnvironment as _SLURMEnvironment  # type: ignore
        except Exception:
            _SLURMEnvironment = None

    slurm_plugin = None
    if _SLURMEnvironment is not None:
        try:
            slurm_plugin = _SLURMEnvironment(auto_requeue=False)
        except TypeError:
            slurm_plugin = _SLURMEnvironment()

    def _slurm_detect() -> bool:
        if slurm_plugin is None:
            return False
        detect = getattr(slurm_plugin, "detect", None)
        if callable(detect):
            try:
                return bool(detect())
            except Exception:
                pass
        detect_cls = getattr(_SLURMEnvironment, "detect", None)
        if callable(detect_cls):
            try:
                return bool(detect_cls())
            except Exception:
                pass
        return "SLURM_JOB_ID" in os.environ

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()
    loggers = [pl.loggers.CSVLogger(save_dir=".")]
    if cfg.use_wandb:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=".",
                name=f"{cfg.data}_{now}",
                project="ManiFM",
                log_model=False,
                config=cfg_dict,
                resume=True,
            )
        )
    trainer = pl.Trainer(
        max_steps=cfg.optim.num_iterations,
        accelerator="gpu",
        devices=1,
        logger=loggers,
        val_check_interval=cfg.val_every,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        plugins=slurm_plugin if _slurm_detect() else None,
        num_sanity_val_steps=0,
    )

    # If we specified a checkpoint to resume from, use it
    checkpoint = cfg.get("resume", None)

    # Check if a checkpoint exists in this working directory.  If so, then we are resuming from a pre-emption
    # This takes precedence over a command line specified checkpoint
    checkpoints = glob("checkpoints/**/*.ckpt", recursive=True)
    if len(checkpoints) > 0:
        # Use the checkpoint with the latest modification time
        checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]

    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)

    train_metrics = trainer.callback_metrics

    log.info("Starting testing!")
    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None
    trainer.test(model, test_loader, ckpt_path=ckpt_path)
    log.info(f"Best ckpt path: {ckpt_path}")

    if cfg.get("delete_checkpoints_after_use", False):
        for p in glob("checkpoints/**/*.ckpt", recursive=True):
            try:
                os.remove(p)
            except Exception:
                pass

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    for k, v in metric_dict.items():
        metric_dict[k] = float(v)

    with open("metrics.json", "w") as fout:
        print(json.dumps(metric_dict), file=fout)

    return metric_dict


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print(traceback.format_exc())
        sys.exit(1)
