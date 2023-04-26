# download a file from wandb
# Usage: python download_wandb_file.py <run_id> <file_name> <output_path>

import argparse
from pathlib import Path

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help="wandb run id")
    parser.add_argument("file_name", type=str, help="file name")
    parser.add_argument("output_path", type=str, help="output path")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.run_id)
    file = run.file(args.file_name)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file.download(replace=True, root=args.output_path)
