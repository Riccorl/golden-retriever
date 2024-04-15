# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""
from functools import partial
from glob import glob
import json
import os
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import datasets as hf_datasets
import psutil
from streaming import MDSWriter, JSONWriter
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# from goldenretriever.common.hf_utils import build_tokenizer
from goldenretriever.data.base.datasets import IterableBaseDataset

WRITER_MAP = {"mds": MDSWriter, "json": JSONWriter}


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="Convert dataset into MDS format, optionally tokenizing"
    )
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument(
        "--data_subset", type=str, default=None, help='E.g. "all" or "en"'
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
    )
    parser.add_argument("--out_root", type=str, required=False)
    parser.add_argument("--compression", type=str, default=None)

    parser.add_argument("--tokenizer", type=str, required=False, default=None)
    parser.add_argument("--tokenizer_kwargs", type=str, required=False)
    parser.add_argument("--num_workers", type=int, required=False, default=None)
    parser.add_argument("--is_local", default=False, action="store_true")
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--writer", type=str, default="json")

    parsed = parser.parse_args()

    if parsed.tokenizer_kwargs is not None:
        parsed.tokenizer_kwargs = json.loads(parsed.tokenizer_kwargs)
    else:
        parsed.tokenizer_kwargs = {}

    return parsed


def build_hf_dataset(
    dataset_name: str,
    split: str,
    data_subset: Union[str, None] = None,
    streaming: bool = True,
    shuffle: bool = False,
    seed: int = 42,
    is_local: bool = False,
    num_workers: Optional[int] = None,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    if is_local:
        if os.path.isdir(dataset_name):
            # only jsonl for now
            data_files = glob(f"{dataset_name}/*.jsonl")
        else:
            data_files = dataset_name
        hf_dataset = hf_datasets.load_dataset(
            "json",
            data_files=data_files,
            split=split,
            streaming=streaming,
            num_proc=num_workers if not streaming else None,
        )
    else:
        hf_dataset = hf_datasets.load_dataset(
            path=dataset_name, name=data_subset, split=split, streaming=streaming
        )
    if shuffle:
        print("Shuffling dataset")
        hf_dataset = hf_dataset.shuffle(seed=seed)
    dataset = IterableBaseDataset(name="hf_data", data=hf_dataset)
    return dataset


def build_dataloader(
    dataset: Dataset, batch_size: int, num_workers: Optional[int]
) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if "linux" or "macos" in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    if batch_size is not None:
        prefetch_factor = (
            max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2
        )
    else:
        prefetch_factor = 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
    loader: DataLoader, tokenizer: PreTrainedTokenizerBase | None = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}

    Yields:
        Sample dicts.
    """
    for batch in loader:
        # print(batch)
        # exit()
        yield batch
        # keys = list(batch.keys())
        # current_bs = len(batch[keys[0]])
        # for idx in range(current_bs):
        #     yield {k: v[idx] if isinstance(v, list) else v for k, v in batch.items()}


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """

    tokenizer = None
    # if args.tokenizer is not None:
    #     tokenizer = build_tokenizer(args.tokenizer, args.tokenizer_kwargs)

    from litdata import optimize

    # 1. Define a function to convert the text within the parquet files into tokens
    # def tokenize_fn(filepath, tokenizer=None):
    #     parquet_file = pq.ParquetFile(filepath)
    #     # Process per batch to reduce RAM usage
    #     for batch in parquet_file.iter_batches(batch_size=8192, columns=["content"]):
    #         for text in batch.to_pandas()["content"]:
    #             yield tokenizer.encode(text, bos=False, eos=True)

    # data_files = glob(f"{args.dataset}/*.jsonl")
    # inputs = []
    with open(args.dataset, "r") as f:
        inputs = [json.loads(line) for line in f.readlines()]
    optimize(
        fn=lambda x: x,
        inputs=inputs,  # Provide any inputs. The fn is applied on each item.
        output_dir="/home/ric/Projects/golden-retriever/data/dpr-like/el/litdata/val",  # The directory where the optimized data are stored.
        num_workers=1,  # The number of workers. The inputs are distributed among them.
        chunk_size=10_000,  # The number of inputs per chunk.
        # chunk_bytes="64MB",  # The maximum number of bytes to write into a data chunk.
    )

    # for split_name in args.splits:

    #     # Get samples
    #     dataset = build_hf_dataset(
    #         dataset_name=args.dataset,
    #         data_subset=args.data_subset,
    #         split=split_name,
    #         shuffle=args.shuffle,
    #         is_local=args.is_local,
    #         num_workers=args.num_workers,
    #     )
    #     loader = build_dataloader(
    #         dataset=dataset, batch_size=None, num_workers=args.num_workers
    #     )
    #     # get columns from iterable dataset
    #     # columns = next(dataset.__iter__()).keys()
    #     columns = {
    #         "id": "str",
    #         "doc_topic": "str",
    #         "question": "str",
    #         "answers": "str",
    #         "positive_ctxs": "json",
    #         "negative_ctxs": "json",
    #         "hard_negative_ctxs": "json",
    #     }

    #     samples = generate_samples(loader, tokenizer=tokenizer)
    #     # reset the generator
    #     # samples = generate_samples(loader, tokenizer=tokenizer)

    #     # Write samples
    #     writer = WRITER_MAP[args.writer]
    #     print(f"Converting {args.dataset} to {writer.__name__} format.")
    #     if args.overwrite:
    #         if os.path.exists(os.path.join(args.out_root, split_name)):
    #             # delete the existing files in the output directory
    #             for file_name in os.listdir(os.path.join(args.out_root, split_name)):
    #                 os.remove(os.path.join(args.out_root, split_name, file_name))

    #     with writer(
    #         columns=columns,
    #         out=os.path.join(args.out_root, split_name),
    #         # compression=args.compression,
    #     ) as out:
    #         # we also want to count the number of tokens for the progress bar
    #         try:
    #             for sample in tqdm(samples, desc=f"Converting {args.dataset} to MDS"):
    #                 out.write(sample)
    #         except Exception as e:
    #             print(f"Exception: {e}")

    #     print(f"Finished converting {split_name} to {writer.__name__} format.")
    # print(f"Finished converting all splits to {writer.__name__} format.")


if __name__ == "__main__":
    main(parse_args())
