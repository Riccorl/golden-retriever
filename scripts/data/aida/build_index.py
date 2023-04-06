import argparse
import os
from pathlib import Path
from typing import Union

import torch

from golden_retriever import GoldenRetriever


@torch.no_grad()
def build_index(
    retriever_name_or_path: Union[str, os.PathLike],
    candidates_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    batch_size: int = 512,
    device: str = "cuda",
):
    retriever = GoldenRetriever.from_pretrained(
        retriever_name_or_path,
        device=device,
    )
    retriever.eval()

    # retriever = torch.compile(retriever, backend="tensorrt")

    with open(candidates_path) as f:
        candidates = [l.strip() for l in f.readlines()]

    retriever.index(
        candidates,
        batch_size=batch_size,
        force_reindex=True,
        context_max_length=64,
        move_index_to_cpu=True,
    )

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    retriever.save_pretrained(output_path)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--retriever_name_or_path", type=str, required=True)
    arg_parser.add_argument("--candidates_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=512)
    arg_parser.add_argument("--device", type=str, default="cuda")

    build_index(**vars(arg_parser.parse_args()))
