import json
import argparse
from pathlib import Path
from random import shuffle

def split_data(
    input_file,
    output_folder,
):
    with open(input_file) as f:
        data = [json.loads(line) for line in f]

    # random select 1M documents and another 1M documents + 1K for dev
    print(data[0])
    shuffle(data)
    print(data[0])
    first_million = data[:1_000_000]
    second_million = data[1_000_000:2_000_000]
    dev = data[2_000_000:2_001_000]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "first_1M.jsonl", "w") as f:
        for sample in first_million:
            f.write(json.dumps(sample) + "\n")
    with open(output_folder / "second_1M.jsonl", "w") as f:
        for sample in second_million:
            f.write(json.dumps(sample) + "\n")
    with open(output_folder / "val.jsonl", "w") as f:
        for sample in dev:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_folder", type=str)

    split_data(**vars(arg_parser.parse_args()))
