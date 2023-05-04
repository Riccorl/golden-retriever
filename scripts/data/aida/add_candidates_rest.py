# Path: scripts/data/aida/add_candidates.py

import argparse
import json
import os
from pathlib import Path
from typing import Union

import requests
import tqdm


def compute_retriever_stats(dataset) -> None:
    correct, total = 0, 0
    for sample in dataset:
        window_candidates = sample["window_candidates"]
        window_candidates = [c.replace("_", " ").lower() for c in window_candidates]

        for ss, se, label in sample["window_labels"]:
            if label == "--NME--":
                continue
            if label.replace("_", " ").lower() in window_candidates:
                correct += 1
            total += 1

    recall = correct / total
    print("Recall:", recall)


def batch_generation(samples, batch_size):
    batch = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def add_candidates(
    endpoint: str,
    input_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    batch_size: int = 512,
    topics: bool = False,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents_batch = []

    output_data = []
    with open(input_path) as f:
        samples = [json.loads(line) for line in f.readlines()]

        for documents_batch in tqdm.tqdm(batch_generation(samples, batch_size)):
            request_data = {"documents": [d["text"] for d in documents_batch]}
            topics_pair = None
            if topics:
                topics_pair = [d["doc_topic"] for d in documents_batch]
                request_data["document_topics"] = topics_pair
            retriever_outs = requests.post(endpoint, json=request_data).json()
            for i, sample in enumerate(documents_batch):
                candidate_titles = [
                    c["label"].split(" <def>", 1)[0] for c in retriever_outs[i]
                ]
                sample["window_candidates"] = candidate_titles
                sample["window_candidates_scores"] = [
                    c["score"] for c in retriever_outs[i]
                ]
                output_data.append(sample)

    with open(output_path, "w") as f:
        for sample in output_data:
            f.write(json.dumps(sample) + "\n")

    # measure some metrics
    compute_retriever_stats(output_data)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/api/retrieve",
    )
    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--topics", action="store_true")

    add_candidates(**vars(arg_parser.parse_args()))
