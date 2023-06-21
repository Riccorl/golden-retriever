import json

import json
import argparse
from pathlib import Path
from random import shuffle
from tqdm import tqdm

def search_data(
    input_file,
    input_dpr_file,
    output_file,
):
    print("loading dpr data")
    with open(input_dpr_file) as f:
        dpr_data = [json.loads(line) for line in f]

    print("Building question set")
    question_set = set([d["question"] for d in dpr_data])
    id_set = set([d["id"] for d in dpr_data])

    found_data = []
    # print("loading data")
    with open(input_file) as f:
        # data = (json.loads(line) for line in f)
    
        for line in tqdm(f, desc="searching"):
            sample = json.loads(line)
            # if sample["text"] in question_set:
            if f"{sample['doc_id']}_{sample['offset']}" in id_set:
                found_data.append(sample)

    print(f"Found {len(found_data)} samples")
    print(f"Saving to {output_file}")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for sample in found_data:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("input_dpr_file", type=str)
    arg_parser.add_argument("output_file", type=str)

    search_data(**vars(arg_parser.parse_args()))
