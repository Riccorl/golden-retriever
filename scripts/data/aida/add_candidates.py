# Path: scripts/data/aida/add_candidates.py

import argparse
import passagelib
import json
import logging
import os
from pathlib import Path
from typing import Union

import torch
import tqdm

from goldenretriever import GoldenRetriever
from goldenretriever.common.log import get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.base.datasets import BaseDataset

logger = get_logger(level=logging.INFO)


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


@torch.no_grad()
def add_candidates(
    retriever_name_or_path: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    batch_size: int = 128,
    num_workers: int = 4,
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

    logger.info(f"Loading from {input_path}")
    with open(input_path) as f:
        samples = [json.loads(line) for line in f.readlines()]

    topics = topics and "doc_topic" in samples[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        # get tokenizer
        tokenizer = retriever.question_tokenizer
        collate_fn = lambda batch: ModelInputs(
            tokenizer(
                [b["text"] for b in batch],
                text_pair=[b["doc_topic"] for b in batch] if topics else None,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
        )
        logger.info(f"Creating dataloader with batch size {batch_size}")
        dataloader = torch.utils.data.DataLoader(
            BaseDataset(name="passage", data=samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        # we also dump the candidates to a file after a while
        retrieved_accumulator = []
        with torch.inference_mode():
            num_completed_docs = 0

            for documents_batch in tqdm.tqdm(dataloader):
                retrieve_kwargs = {
                    **documents_batch,
                    "k": 100,
                    "precision": precision,
                }
                batch_out = retriever.retrieve(**retrieve_kwargs)
                retrieved_accumulator.extend(batch_out)

                if len(retrieved_accumulator) % 300_000 == 0:
                    output_data = []
                    # get the correct document from the original dataset
                    # the dataloader is not shuffled, so we can just count the number of
                    # documents we have seen so far
                    for sample, retrieved in zip(
                        samples[
                            num_completed_docs : num_completed_docs
                            + len(retrieved_accumulator)
                        ],
                        retrieved_accumulator,
                    ):
                        candidate_titles = [
                            c.label.split(" <def>", 1)[0] for c in retrieved
                        ]
                        sample["window_candidates"] = candidate_titles
                        sample["window_candidates_scores"] = [
                            c.score for c in retrieved
                        ]
                        output_data.append(sample)

                    for sample in output_data:
                        f_out.write(json.dumps(sample) + "\n")

                    num_completed_docs += len(retrieved_accumulator)
                    retrieved_accumulator = []

            if len(retrieved_accumulator) > 0:
                output_data = []
                # get the correct document from the original dataset
                # the dataloader is not shuffled, so we can just count the number of
                # documents we have seen so far
                for sample, retrieved in zip(
                    samples[
                        num_completed_docs : num_completed_docs
                        + len(retrieved_accumulator)
                    ],
                    retrieved_accumulator,
                ):
                    candidate_titles = [
                        c.label.split(" <def>", 1)[0] for c in retrieved
                    ]
                    sample["window_candidates"] = candidate_titles
                    sample["window_candidates_scores"] = [c.score for c in retrieved]
                    output_data.append(sample)

                for sample in output_data:
                    f_out.write(json.dumps(sample) + "\n")

                num_completed_docs += len(retrieved_accumulator)
                retrieved_accumulator = []

    # compute_retriever_stats(samples)


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
