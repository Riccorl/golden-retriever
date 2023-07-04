# Path: scripts/data/aida/add_candidates.py

import argparse
import passagelib
import json
import logging
import os
from pathlib import Path
from typing import List, Union
import requests

import torch
import tqdm

from goldenretriever import GoldenRetriever
from goldenretriever.common.log import get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.base.datasets import BaseDataset

logger = get_logger(level=logging.INFO)


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
):
    logger.info(f"Loading from {input_path}")
    with open(input_path) as f:
        data = [json.loads(line) for line in f.readlines()]

    logger.info(f"Retrieving candidates for {len(data)} documents")

    logger.info(f"Writing to {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    document_path = output_path / "documents_windowed_candidates.jsonl"
    summary_path = output_path / "summaries_windowed_candidates.jsonl"

    with open(document_path, "w") as doc_f, open(summary_path, "w") as sum_f:
        for batch in tqdm.tqdm(batch_generation(data, batch_size)):
            response = requests.post(
                endpoint, json=[b["document"] for b in batch]
            ).json()
            # replace the document id in the response with the one from the input
            # the response has the same order as the input and the document ids can
            # be used more than once
            response_doc_ids = [b["doc_id"] for b in response]
            # remove duplicates while preserving order
            response_doc_ids = list(dict.fromkeys(response_doc_ids))
            # now map the response doc ids to the batch doc ids
            doc_id_map = {
                response_doc_ids[i]: batch[i]["id"]
                for i in range(len(response_doc_ids))
            }
            # now replace the doc ids in the response with the ones from the input
            for i in range(len(response)):
                response[i]["doc_id"] = doc_id_map[response[i]["doc_id"]]

            # documents_out.extend(response)
            for sample in response:
                doc_f.write(json.dumps(sample) + "\n")

            # do the same for the summaries
            response = requests.post(
                endpoint, json=[b["summary"] for b in batch]
            ).json()
            # replace the document id in the response with the one from the input
            # the response has the same order as the input and the document ids can
            # be used more than once
            response_doc_ids = [b["doc_id"] for b in response]
            # remove duplicates while preserving order
            response_doc_ids = list(dict.fromkeys(response_doc_ids))
            # now map the response doc ids to the batch doc ids
            doc_id_map = {
                response_doc_ids[i]: batch[i]["id"]
                for i in range(len(response_doc_ids))
            }
            # now replace the doc ids in the response with the ones from the input
            for i in range(len(response)):
                response[i]["doc_id"] = doc_id_map[response[i]["doc_id"]]

            for sample in response:
                sum_f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/api/gerbil",
    )
    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=128)

    add_candidates(**vars(arg_parser.parse_args()))
