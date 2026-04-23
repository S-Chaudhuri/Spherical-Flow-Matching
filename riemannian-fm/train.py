"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os

# Use PyTorch backend for geomstats
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import os.path as osp
import sys
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

from codecarbon import EmissionsTracker # carbon tracking using CodeCarbon

from manifm.datasets import get_loaders
from manifm.model_pl import ManifoldFMLitModule



torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)


def make_emissions_tracker():           # helper function for carbon tracking initialization
    return EmissionsTracker(
        project_name = "geometry-and-RFM",
        output_dir = ".",
        output_file = "emissions.csv",
        save_to_file = True,
        log_level = "error",
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    logging.getLogger("pytorch_lightning").setLevel(logging.getLevelName("INFO"))

    if cfg.get("seed", None) is not None:
        #pl.utilities.seed.seed_everything(cfg.seed)
        pl.seed_everything(cfg.seed)

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
                log_model=True,
                config=cfg_dict,
                resume=True,
            )
        )
    trainer = pl.Trainer(
        max_steps=cfg.optim.num_iterations,
                                        # also allow to run on cpu
        accelerator = cfg.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu"),
                                        # allow number of devices to be set through the config
        devices = cfg.get("devices", 1),
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

    tracker = make_emissions_tracker()  # init carbon tracking using make_emissions_tracker() helper function
    tracker.start()                     # start carbon tracking
                                        # source: https://wandb.ai/amanarora/codecarbon/reports/Tracking-CO2-Emissions-of-
                                        # Your-Deep-Learning-Models-with-CodeCarbon-and-Weights-Biases--VmlldzoxMzM1NDg3
    try:
        trainer.fit(model, train_loader, val_loader, ckpt_path = checkpoint)
    finally:                            # also save in case training fails
        train_emissions_kg = tracker.stop()
    train_metrics = trainer.callback_metrics

    log.info("Starting testing!")
    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None
    trainer.test(model, test_loader, ckpt_path=ckpt_path)
    log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    for k, v in metric_dict.items():
        if torch.is_tensor(v):          # type check for tensors and detach if so
            metric_dict[k] = v.detach().cpu().item()
        else:
            metric_dict[k] = float(v)
                                        # also add total CO2kq equivalent to the metric dict
    metric_dict["train/codecarbon_kg_co2"] = float(train_emissions_kg)

    with open("metrics.json", "w") as fout:
        print(json.dumps(metric_dict), file = fout)

    return metric_dict


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print(traceback.format_exc())
        sys.exit(1)
