import json
import argparse
from pathlib import Path


def split_data(
    input_file,
    output_folder,
    train_size=0.8,
    dev_size=0.1,
    test_size=0.1,
):
    with open(input_file) as f:
        data = json.load(f)
    
    # keep some statistics
    num_docs = len(data)
    # filter out documents with no text
    data = [doc for doc in data if len(doc["question"]) > 0]
    num_docs_with_text = len(data)
    print(f"Number of documents: {num_docs}")
    print(f"Number of documents with text: {num_docs_with_text}")
    # check the sum of the sizes
    assert train_size + dev_size + test_size == 1

    train_size = int(len(data) * train_size)
    dev_size = int(len(data) * dev_size)
    test_size = int(len(data) * test_size)

    train_data = data[:train_size]
    dev_data = data[train_size : train_size + dev_size]
    test_data = data[train_size + dev_size :]

    print(
        f"Train size: {len(train_data)}\nDev size: {len(dev_data)}\nTest size: {len(test_data)}"
    )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open(output_folder / "val.json", "w") as f:
        json.dump(dev_data, f, indent=2)
    with open(output_folder / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_folder", type=str)
    arg_parser.add_argument("--train_size", type=float, default=0.8)
    arg_parser.add_argument("--dev_size", type=float, default=0.1)
    arg_parser.add_argument("--test_size", type=float, default=0.1)

    split_data(**vars(arg_parser.parse_args()))
