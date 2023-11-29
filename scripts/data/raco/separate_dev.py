import argparse
import json


def main(input_path, output_folder):
    with open(input_path, "r") as f:
        data = json.load(f)

    datasets = {}
    for d in data:
        dataset_name = d["dataset"]
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(d)

    for dataset_name, dataset in datasets.items():
        with open(f"{output_folder}/{dataset_name}.json", "w") as f:
            json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--output_folder", type=str, required=True)

    main(**vars(arg_parser.parse_args()))
