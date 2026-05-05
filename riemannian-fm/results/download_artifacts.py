import argparse
import os

import torch
import wandb
from tqdm import tqdm


def main():
    # fmt: off
    # 1. Set up Argument Parsing
    parser = argparse.ArgumentParser(description="Download and inspect W&B artifacts for Geometric Flow Matching.")

    # Required/Standard W&B Arguments
    parser.add_argument("--entity", "-e", default="floris-de-kam-university-of-amsterdam", help="W&B username or team name (e.g., your teammate's username)")
    parser.add_argument("--project", "-p", default="ManiFM", help="W&B project name (default: ManiFM)")
    parser.add_argument("--artifact", "-a", default="final_fixed_eval_outputs", help="Name of the artifact (default: final_fixed_eval_outputs)")
    parser.add_argument("--save_dir", "-s", default="./results/artifacts", help="Local directory to save the artifacts (default: ./results/artifacts)")

    # Version Arguments (User MUST provide either a single version or a range)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", "-v", type=int, help="Single version index to download (e.g., 20 for 'v20')")
    group.add_argument("--range", "-r", type=int, nargs=2, metavar=("START", "END"), help="Start and end indices for a range of versions (inclusive, e.g., 0 10)")

    args = parser.parse_args()
    # fmt: on

    # 2. Determine which versions to download based on arguments
    if args.version is not None:
        versions_to_download = [args.version]
        print(f"Targeting single version: v{args.version}")

    else:
        start, end = args.range
        versions_to_download = list(range(start, end + 1))
        print(f"Targeting version range: v{start} to v{end} ({len(versions_to_download)} total)")

    # 3. Authenticate and Initialize API
    wandb.login()
    api = wandb.Api()

    # 4. Download Loop with Progress Bar
    successful_downloads = []

    # Wrap the version list in tqdm for the progress bar
    pbar = tqdm(versions_to_download, desc="Downloading Artifacts", unit="artifact")

    for v in pbar:
        artifact_path = f"{args.entity}/{args.project}/{args.artifact}:v{v}"

        # Create a version-specific subfolder so files don't overwrite each other
        v_save_dir = os.path.join(args.save_dir, f"v{v}")
        os.makedirs(v_save_dir, exist_ok=True)

        try:
            # Update progress bar description to show current action
            pbar.set_postfix({"Current": f"v{v}"})

            # Fetch and download the artifact
            artifact = api.artifact(artifact_path)

            # Iterate through files and force-download to bypass symlink issues
            files = list(artifact.files())

            if len(files) == 0:
                tqdm.write(f"\n[Warning] Artifact v{v} exists but contains no files!")
                continue

            for file in files:
                file.download(root=v_save_dir, replace=True)

            successful_downloads.append(v_save_dir)

        except Exception as e:
            # Use tqdm.write so it doesn't break the visual progress bar
            tqdm.write(f"\n[Warning] Failed to download v{v}. It might not exist. Error: {e}")

    # 5. Post-Download Verification (Print metadata for the last downloaded item)
    if successful_downloads:
        last_dir = successful_downloads[-1]
        file_path = os.path.join(last_dir, f"{args.artifact}.pt")

        if os.path.exists(file_path):
            n = len(successful_downloads)
            print("\n--- Download Complete ---")
            print(f"Successfully downloaded {n} artifacts to {args.save_dir}")
            print(f"Loading the most recent download ({file_path}) to verify...")

            try:
                results = torch.load(file_path, map_location="cpu", weights_only=True)
                if "meta" in results:
                    print("Verification successful! Run Metadata:")
                    for key, value in results["meta"].items():
                        print(f"  {key}: {value}")

            except Exception as e:
                print(f"Artifact downloaded, but failed to load PyTorch file: {e}")

    else:
        error_msg = "\nNo artifacts were successfully downloaded. Please check your entity name and version indices."
        print(error_msg)


if __name__ == "__main__":
    main()
