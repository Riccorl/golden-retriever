import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import hydra
import torch
import transformers as tr
from omegaconf import omegaconf
from rich.progress import track

from data.pl_data_modules import BasePLDataModule
from models.pl_modules import BasePLModule
from utils.logging import get_console_logger

logger = get_console_logger()


def batch_generator(inputs: Sequence, batch_size: int) -> Sequence:
    """
    Batch generator for neural models.
    """
    batch = []
    for x in inputs:
        batch.append(inputs[x])
        if len(batch) >= batch_size:
            yield batch
            batch = []
    # yield leftovers
    if batch:
        yield batch


@torch.no_grad()
def produce_context_embeddings(
    checkpoint_path: Union[str, os.PathLike],
    data: Any,
    batch_size: int = 32,
    device: str = "cuda",
):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"Using {device} as device")

    # pl_data_module: BasePLDataModule = hydra.utils.instantiate(
    #     conf.data.datamodule, _recursive_=False
    # )
    # pl_data_module.prepare_data()
    # pl_data_module.setup("test")

    logger.log(f"Instantiating the Model from {checkpoint_path}")
    pl_module = BasePLModule.load_from_checkpoint(checkpoint_path, _recursive_=False)
    pl_module.to(device)
    pl_module.eval()

    # get the tokenizer
    tokenizer = tr.AutoTokenizer.from_pretrained(
        pl_module.model.context_encoder.name_or_path
    )

    # do stuff
    contex_encoder = pl_module.model.context_encoder
    context_embeddings = []
    embedding_dict = {x["definition"]: i for i, x in enumerate(data.values())}
    for batch in track(batch_generator(data, batch_size)):
        batch = [x["definition"] for x in batch]
        batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        output = contex_encoder(**batch).pooler_output

        context_embeddings.append(output)

    context_embeddings = torch.cat(context_embeddings, dim=0)
    return context_embeddings, embedding_dict


def main(
    checkpoint_path: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike],
    output_path: Optional[Union[str, os.PathLike]],
    batch_size: int = 32,
    device: str = "cuda",
):
    # load the data
    with open(input_path, "r") as f:
        data = json.load(f)
    
    context_embeddings, embedding_dict = produce_context_embeddings(
        checkpoint_path=checkpoint_path,
        data=data,
        batch_size=batch_size,
        device=device,
    )
    # save the embeddings to disk
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(context_embeddings, output_path / "context_embeddings.pt")
    with open(output_path / "embedding_dict.json", "w") as f:
        json.dump(embedding_dict, f, indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--checkpoint", type=str, required=True)
    arg_parser.add_argument("--input", type=str, required=True)
    arg_parser.add_argument("--output", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--device", type=str, default="cuda")
    args = arg_parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
