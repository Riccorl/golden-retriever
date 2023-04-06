# Path: scripts/data/aida/add_candidates.py

import argparse
import json
import os
from typing import Union

import torch
import tqdm

from golden_retriever import GoldenRetriever


@torch.no_grad()
def add_candidates(
    retriever_name_or_path: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    batch_size: int = 512,
    device: str = "cuda",
):
    retriever = GoldenRetriever.from_pretrained(
        retriever_name_or_path,
        device=device,
    )
    retriever.eval()

    documents_batch = []
    with open(output_path, "w") as f:
        for line in tqdm.tqdm(open(input_path)):
            sample = json.loads(line)
            documents_batch.append(sample)
            if len(documents_batch) == batch_size:
                candidates = retriever.retrieve(
                    [d["text"] for d in documents_batch], k=100
                )
                for i, sample in enumerate(documents_batch):
                    candidate_titles = [
                        c.split(" <def>", 1)[0] for c in candidates[0][i]
                    ]
                    sample["window_candidates"] = candidate_titles
                    f.write(json.dumps(sample) + "\n")
                documents_batch = []

        if len(documents_batch) > 0:
            candidates = retriever.retrieve([d["text"] for d in documents_batch], k=100)
            for i, sample in enumerate(documents_batch):
                candidate_titles = [c.split(" <def>", 1)[0] for c in candidates[0][i]]
                sample["window_candidates"] = candidate_titles
                f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--retriever_name_or_path", type=str, required=True)
    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=512)
    arg_parser.add_argument("--device", type=str, default="cuda")

    add_candidates(**vars(arg_parser.parse_args()))
