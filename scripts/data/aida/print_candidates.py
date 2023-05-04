# Path: scripts/data/aida/add_candidates.py

import argparse
import json
import os
from pathlib import Path
from typing import Union


def add_candidates(
    input_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as w:
        with open(input_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                candidates = [c.replace(" ", "_") for c in sample["window_candidates"]]
                labels = [l[-1] for l in sample["window_labels"]]

                # w.write(f"{sample['text']}\n")
                for candidate in candidates:
                    if candidate in labels:
                        w.write(f"{candidate} **\n")
                    else:
                        w.write(f"{candidate}\n")
                w.write("\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)

    add_candidates(**vars(arg_parser.parse_args()))
