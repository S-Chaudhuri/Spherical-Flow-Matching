# Geometric Flow Matching: Data Download Guide

This guide explains how to download pre-computed model evaluation data (artifacts) from Weights & Biases (W&B) to your local machine for analysis. 

W&B is a cloud-based machine learning platform where our training runs and evaluation outputs are securely stored. You do not need to be an expert in W&B to download the data—just follow the steps below!

---

## 🛠 Step 1: Prerequisites

Before running the download script, ensure you have Python installed, along with the required libraries. Open your terminal or command prompt and run:

```bash
pip install wandb torch tqdm
```

---

## 🔐 Step 2: Authentication (Your API Key)
To download private data, W&B needs to know you have permission. You will do this using an **API Key**.
1. Go to [wandb.ai](https://wandb.ai/site/) in your web browser.
2. Log in using the team credentials:
    - **Email:** floris.de.kam@student.uva.nl
    - **Password:** DL2warriors123#
3. Once logged in, navigate to [wandb.ai/authorize](https://wandb.ai/authorize).
4. Click "Generate new API key" and copy the long string of text (your API Key) displayed on the screen.

Note: The first time you run the download script, your terminal will pause and ask you to choose between 3 options. Choose the second (or third I don't remember) option and paste this API key. Once you paste it and hit Enter, it will save the key locally so you don't have to do it again.

---

## 🚀 Step 3: How to Use the Script
The download script (`download_artifacts.py`) is run from your terminal. You control what it downloads by passing "arguments" (flags).

### Required Arguments
You must specify either a single version or a range of versions to download.
- `--version` (or `-v`): Use this to download a single artifact version (e.g., `20` for version 20).
- `--range` (or `-r`): Use this to download multiple versions. Provide the start and end numbers separated by a space (e.g., `0 10`).

### Optional Arguments (Defaults)
If you don't provide these, the script uses the defaults set up for our specific codebase.  
- `--entity` (or `-e`): The W&B username or team workspace name where the project is stored. Default is `floris-de-kam-university-of-amsterdam`. 
- `--project` (or `-p`): The W&B project name. Default is `ManiFM`. 
- `--artifact` (or `-a`): The name of the saved data. Default is `final_fixed_eval_outputs`.  
- `--save_dir` (or `-s`): Where to save the files on your computer. Default is `./artifacts.`

---

## 📖 Usage Examples
### Example 1: Downloading a Single Version
To download the results of experiment 20 (v20), run this command:
```bash
python download_artifacts.py --version 20
```

### Example 2: Downloading a Range of Versions
If you want to analyze versions 1 through 5 in a batch:
```bash
python download_artifacts.py --range 1 5
```

### Example 3: Saving to a Custom Folder
If you want to keep your workspace organized and save the data to a specific folder like `my_analysis_data`:
```bash
python download_artifacts.py --range 1 5 --save_dir ./my_analysis_data
```

---

## 📂 Where is my data?
By default, the script creates an `artifacts` folder in the same directory where you ran the script. To prevent files from overwriting each other, it creates sub-folders for every version.

Your directory will look like this:
```text
📁 artifacts/
 ├── 📁 v1/
 │    └── final_fixed_eval_outputs.pt
 ├── 📁 v2/
 │    └── final_fixed_eval_outputs.pt
 └── 📁 v20/
      └── final_fixed_eval_outputs.pt
```

### How to open the data in Python / Jupyter Notebook
Once downloaded, you can load the data into a Python script or Jupyter Notebook using PyTorch:
```python
import torch

# Load the data for version 20
data = torch.load("./artifacts/v20/final_fixed_eval_outputs.pt", map_location="cpu", weights_only=True)

# View what is inside
print(data.keys())
print("Metadata:", data["meta"])
```