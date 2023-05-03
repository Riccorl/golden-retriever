# Path: scripts/data/aida/add_candidates.py

import argparse
import json
import os
from pathlib import Path
from typing import Union

import torch
import tqdm

from golden_retriever import GoldenRetriever


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

    # doc_level_correct, doc_level_total = 0, 0
    # for sample in dataset:
    #     doc_id = sample["doc_id"]


def batch_generation(samples, batch_size):
    batch = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


@torch.no_grad()
def add_candidates(
    retriever_name_or_path: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    batch_size: int = 512,
    device: str = "cuda",
    index_device: str = "cpu",
    precision: str = "fp32",
    faiss: bool = False,
    topics: bool = False,
):
    retriever = GoldenRetriever.from_pretrained(
        retriever_name_or_path,
        device=device,
        index_device=index_device,
        index_precision=precision,
        load_faiss_index=faiss,
    )
    retriever.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents_batch = []

    debug_retrieval_stuff = {}

    output_data = []
    with open(input_path) as f:
        samples = [json.loads(line) for line in f.readlines()]

    with torch.inference_mode():
        for documents_batch in tqdm.tqdm(
            batch_generation(samples, batch_size)
        ):
            topics_pair = None
            if topics:
                topics_pair = [d["doc_topic"] for d in documents_batch]
            retriever_outs = retriever.retrieve(
                [d["text"] for d in documents_batch],
                text_pair=topics_pair,
                k=100,
                precision=precision,
            )
            for i, sample in enumerate(documents_batch):
                candidate_titles = [
                    c.label.split(" <def>", 1)[0] for c in retriever_outs[i]
                ]
                sample["window_candidates"] = candidate_titles
                sample["window_candidates_scores"] = [
                    c.score for c in retriever_outs[i]
                ]
                output_data.append(sample)
                debug_retrieval_stuff[f"{sample['doc_id']}_{sample['window_id']}"] = [
                    (c.label.split(" <def>", 1)[0], c.score) for c in retriever_outs[i]
                ]

    with open(output_path, "w") as f:
        for sample in output_data:
            f.write(json.dumps(sample) + "\n")

    # measure some metrics
    compute_retriever_stats(output_data)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--retriever_name_or_path", type=str, required=True)
    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--index_device", type=str, default="cpu")
    arg_parser.add_argument("--precision", type=str, default="fp32")
    arg_parser.add_argument("--faiss", action="store_true")
    arg_parser.add_argument("--topics", action="store_true")

    add_candidates(**vars(arg_parser.parse_args()))
