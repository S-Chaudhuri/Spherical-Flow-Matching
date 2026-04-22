"""Utilities for running and tracking RFM experiments."""
import yaml
import json
import subprocess
from pathlib import Path
import pandas as pd


def load_experiments(yaml_path):
    """Load an experiment set from a YAML file."""
    return yaml.safe_load(Path(yaml_path).read_text())


def build_cmd(name, cfg, run_dir, seed=42):
    """Build the train.py command for a single experiment."""
    overrides = [f"{k}={v}" for k, v in cfg.items()]
    return [
        "python", "train.py",
        "experiment=general_fm",
        f"seed={seed}",
        f"hydra.run.dir={run_dir}",
        *overrides,
    ]


def run_experiment(name, cfg, set_name, seed=42, skip_if_done=True):
    run_dir = Path(f"runs/{set_name}/{name}").resolve()
    if skip_if_done and (run_dir / "metrics.json").exists():
        print(f"[skip] {set_name}/{name} already done")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(name, cfg, run_dir, seed)
    print(f"[run]  {set_name}/{name}")
    subprocess.run(cmd, check=True, cwd="riemannian-fm")


def run_set(yaml_path, seed=42, skip_if_done=True):
    """Run every experiment in a given YAML set."""
    set_name = Path(yaml_path).stem.replace("experiments_", "")
    experiments = load_experiments(yaml_path)
    for name, cfg in experiments.items():
        run_experiment(name, cfg, set_name, seed, skip_if_done)


# def load_results(yaml_path):
#     """Load metrics + config for every experiment in a set as a DataFrame."""
#     set_name = Path(yaml_path).stem.replace("experiments_", "")
#     experiments = load_experiments(yaml_path)
#     rows = []
#     for name, cfg in experiments.items():
#         metrics_file = Path(f"runs/{set_name}/{name}/metrics.json")
#         row = {"set": set_name, "name": name, **cfg}
#         if metrics_file.exists():
#             row.update(json.loads(metrics_file.read_text()))
#             row["status"] = "done"
#         else:
#             row["status"] = "pending"
#         rows.append(row)
#     return pd.DataFrame(rows)