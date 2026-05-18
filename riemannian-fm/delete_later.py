# import os
# import sys
# import torch
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import concurrent.futures
# from omegaconf import OmegaConf

# # Import the dataset generator from your codebase
# from manifm.datasets import GeneralDataset

# def parse_job_file(file_path):
#     """
#     Parses a bash/slurm script and reconstructs multi-line python commands.
#     Returns a list of extracted Hydra override lists.
#     """
#     if not os.path.exists(file_path):
#         print(f"Error: File '{file_path}' not found.")
#         return []

#     with open(file_path, 'r') as f:
#         lines = f.readlines()

#     commands = []
#     current_cmd = ""
    
#     for line in lines:
#         line = line.strip()
#         # Skip pure comments
#         if line.startswith("#") and not current_cmd:
#             continue

#         # Handle bash line continuations
#         if line.endswith("\\"):
#             current_cmd += line[:-1] + " "
#         else:
#             current_cmd += line
            
#             # If we completed a command that runs train.py, extract it
#             if "python train.py" in current_cmd and not current_cmd.strip().startswith("#"):
#                 # Split by whitespace to get individual arguments
#                 tokens = current_cmd.split()
                
#                 # Filter for Hydra overrides (must contain '=' and shouldn't be the python call itself)
#                 overrides = [t for t in tokens if "=" in t]
#                 commands.append(overrides)
                
#             current_cmd = ""
            
#     return commands


# def visualize_overrides(args):
#     """
#     Worker function to process a single job configuration.
#     """
#     overrides, job_index = args
    
#     run_dir = next((o.split('=')[1] for o in overrides if "hydra.run.dir" in o), f"Job_{job_index+1}")
#     title = os.path.basename(run_dir)
    
#     try:
#         # 1. Parse dotlist overrides into OmegaConf
#         cfg = OmegaConf.from_dotlist(overrides)
        
#         cfg.save_artifacts = False
#         cfg.eval_n_pairs = 100
        
#         if "general" not in cfg:
#             raise ValueError("No 'general' config found.")
            
#         # PRO-TIP: Reduce n_samples for visualization to speed up sampling even more
#         # 1000 points is usually plenty to see the distribution shape.
#         n_vis = 1000 
#         cfg.general.n_samples = n_vis
        
#         if "x0_dist" not in cfg.general:
#             cfg.general.x0_dist = "gaussian"

#         # 2. Instantiate Dataset
#         dataset = GeneralDataset(cfg)
        
#         # 3. Sample Data
#         x0_list, x1_list = [], []
#         for i in range(n_vis):
#             sample = dataset[i]
#             x0_list.append(sample["x0"])
#             x1_list.append(sample["x1"])
            
#         x0 = torch.stack(x0_list).numpy()
#         x1 = torch.stack(x1_list).numpy()
        
#         dim = cfg.general.dim
#         manifold = cfg.general.manifold
        
#         # 4. Plotting
#         fig = plt.figure(figsize=(7, 7))
#         ax = fig.add_subplot(111)
            
#         x0_plot = x0[:, :2]
#         x1_plot = x1[:, :2]

#         if manifold == "poincare":
#             circle = plt.Circle((0, 0), 1.0, color="k", fill=False, linestyle="--", alpha=0.5)
#             ax.add_patch(circle)
#             ax.set_xlim([-1.1, 1.1])
#             ax.set_ylim([-1.1, 1.1])
#         elif manifold == "sphere" and dim == 2:
#             c = cfg.general.curvature
#             R = 1.0 / np.sqrt(c) if c > 0 else 1.0
#             theta = np.linspace(0, 2 * np.pi, 400)
#             ax.plot(R * np.cos(theta), R * np.sin(theta), color="k", linestyle="--", alpha=0.5)
#             ax.set_xlim([-1.25 * R, 1.25 * R])
#             ax.set_ylim([-1.25 * R, 1.25 * R])
            
#         ax.scatter(x0_plot[:, 0], x0_plot[:, 1], s=4, alpha=0.6, color="C0", label="Source (x0)")
#         ax.scatter(x1_plot[:, 0], x1_plot[:, 1], s=4, alpha=0.6, color="C3", label="Target (x1)")
        
#         ax.set_aspect("equal")
#         ax.set_title(title, fontsize=14)
#         ax.legend(loc="upper right")
#         plt.tight_layout()
        
#         os.makedirs("visualizations", exist_ok=True)
#         save_path = os.path.join("visualizations", f"{title}.png")
#         plt.savefig(save_path, dpi=150, bbox_inches='tight') # Reduced DPI from 300 to 150 for speed
#         plt.close(fig)
        
#         return f"✓ [{title}] Done."

#     except Exception as e:
#         return f"❌ [{title}] FAILED. Reason: {type(e).__name__}: {str(e)}"


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python visualize_job_file.py <path_to_job_file>")
#         sys.exit(1)
        
#     job_file = sys.argv[1]
#     commands = parse_job_file(job_file)
    
#     if not commands:
#         print("No valid 'python train.py' commands found in the file.")
#     else:
#         print(f"Found {len(commands)} jobs to process.")
#         tasks = [(overrides, idx) for idx, overrides in enumerate(commands)]
        
#         # Use a ProcessPoolExecutor to run them in parallel
#         # Note: max_workers defaults to the number of processors on your machine
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             # As tasks complete, print their status
#             for result in executor.map(visualize_overrides, tasks):
#                 print(result)

import os
import sys
import torch
from omegaconf import OmegaConf

# Import the model and dataset from your codebase
from manifm.datasets import GeneralDataset
from manifm.model_pl import ManifoldFMLitModule

def parse_job_file(file_path):
    """Extracts Hydra overrides from the .job file."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    commands = []
    current_cmd = ""
    for line in lines:
        line = line.strip()
        if line.startswith("#") and not current_cmd:
            continue
        if line.endswith("\\"):
            current_cmd += line[:-1] + " "
        else:
            current_cmd += line
            if "python train.py" in current_cmd and not current_cmd.strip().startswith("#"):
                tokens = current_cmd.split()
                # Extract Hydra overrides (key=value)
                overrides = [t for t in tokens if "=" in t]
                commands.append(overrides)
            current_cmd = ""
    return commands

def test_forward_pass(overrides, job_index):
    """
    Simulates Hydra's configuration composition, initializes the model and dataset 
    on the CPU, and runs a single forward pass to verify the math.
    """
    print(f"\n{'-'*50}")
    print(f"Testing Network Pass for Job #{job_index + 1}")
    
    run_dir = next((o.split('=')[1] for o in overrides if "hydra.run.dir" in o), f"Job_{job_index+1}")
    title = os.path.basename(run_dir)
    print(f"Title: {title}")
    
    try:
        # 1. Extract the experiment name and clean the overrides
        exp_name = "general_fm" # Default fallback
        clean_overrides = []
        for o in overrides:
            if o.startswith("experiment="):
                exp_name = o.split("=")[1]
            else:
                clean_overrides.append(o)
                
        # 2. Define paths to your base configuration files
        base_cfg_path = os.path.join("configs", "train.yaml")
        exp_cfg_path = os.path.join("configs", "experiment", f"{exp_name}.yaml")
        
        # 3. Load the YAML files using OmegaConf
        if os.path.exists(base_cfg_path):
            base_cfg = OmegaConf.load(base_cfg_path)
        else:
            base_cfg = OmegaConf.create()
            print(f"Warning: Base config not found at {base_cfg_path}")
            
        if os.path.exists(exp_cfg_path):
            exp_cfg = OmegaConf.load(exp_cfg_path)
        else:
            exp_cfg = OmegaConf.create()
            print(f"Warning: Experiment config not found at {exp_cfg_path}")
            
        # 4. Parse the command-line overrides from your job file
        override_cfg = OmegaConf.from_dotlist(clean_overrides)
        
        # 5. MERGE: Replicate Hydra's hierarchy (Base -> Experiment -> Overrides)
        cfg = OmegaConf.merge(base_cfg, exp_cfg, override_cfg)
        
        # 6. Safety overrides for fast CPU testing (don't load/save thousands of points)
        cfg.general.n_samples = 10
        cfg.save_artifacts = False
        cfg.eval_n_pairs = 10
        cfg.use_wandb = False
            
        # 7. Instantiate Dataset (Uses the fully resolved config now!)
        dataset = GeneralDataset(cfg)
        
        # 8. Create a small mock batch (Batch Size = 4)
        x0_list, x1_list = [], []
        for i in range(4):
            sample = dataset[i]
            x0_list.append(sample["x0"])
            x1_list.append(sample["x1"])
            
        batch = {
            "x0": torch.stack(x0_list), # CPU tensor
            "x1": torch.stack(x1_list)  # CPU tensor
        }
        
        # 9. Instantiate Model on CPU
        model = ManifoldFMLitModule(cfg)
        model.eval() # Disable dropout/batchnorm randomness
        
        # 10. The Pseudo Pass: Compute RFM Loss
        with torch.no_grad():
            loss = model.rfm_loss_fn(batch)
            
        # 11. Verify outputs
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ FAILED: Loss is {loss.item()}. Check curvature or dimension settings!")
        else:
            print(f"✓ SUCCESS: Forward pass complete! Computed Loss: {loss.item():.4f}")

    except Exception as e:
        print(f"❌ FAILED to process {title}.")
        print(f"Reason: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_network_pass.py <path_to_job_file>")
        sys.exit(1)
        
    job_file = sys.argv[1]
    commands = parse_job_file(job_file)
    
    if not commands:
        print("No valid configurations found.")
    else:
        for idx, overrides in enumerate(commands[2::3]):
            test_forward_pass(overrides, idx)