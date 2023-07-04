import argparse
import os
from pathlib import Path
from typing import Union

import torch

from goldenretriever import GoldenRetriever


@torch.no_grad()
def build_index(
    retriever_name_or_path: Union[str, os.PathLike],
    candidates_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    batch_size: int = 512,
    passage_max_length: int = 64,
    device: str = "cuda",
    index_device: str = "cpu",
    precision: str = "fp32",
    faiss: bool = False,
):
    retriever = GoldenRetriever.from_pretrained(
        retriever_name_or_path,
        device=device,
        index_device=index_device,
        index_precision=precision,
        load_index_vector=False,
        load_faiss_index=faiss,
    )
    retriever.eval()

    with open(candidates_path) as f:
        candidates = [l.strip() for l in f.readlines()]

    retriever.index(
        candidates,
        batch_size=batch_size,
        force_reindex=True,
        passage_max_length=passage_max_length,
        precision=precision,
        index_precision=precision,
        move_index_to_cpu=bool(index_device == "cpu"),
    )

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    retriever.save_pretrained(output_path)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--retriever_name_or_path", type=str, required=True)
    arg_parser.add_argument("--candidates_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--passage_max_length", type=int, default=64)
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--index_device", type=str, default="cpu")
    arg_parser.add_argument("--precision", type=str, default="fp32")
    arg_parser.add_argument("--faiss", action="store_true")

    build_index(**vars(arg_parser.parse_args()))
