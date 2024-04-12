from glob import glob
import os
from typing import Any, Dict, Iterable

from composer.utils import dist
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import datasets as hf_datasets


def build_tokenizer(
    tokenizer_name: str, tokenizer_kwargs: Dict[str, Any] | None = None
) -> PreTrainedTokenizerBase:

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = tokenizer_kwargs.pop(
        "TOKENIZERS_PARALLELISM", "false"
    )

    signal_file_path = (
        f".node_{dist.get_node_rank()}_local_rank0_completed_tokenizer_setup"
    )

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # Make sure the tokenizer files are downloaded and cached first by local rank 0
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            pass

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get("model_max_length", int(1e30))

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if dist.get_local_rank() == 0:
            with open(signal_file_path, "wb") as f:
                f.write(b"local_rank0_completed_tokenizer_setup")

        dist.barrier()

        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

    return tokenizer


def build_hf_dataset(
    dataset_name: str,
    split: str,
    data_subset: str | None = None,
    streaming: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    is_local: bool = False,
    num_workers: int | None = None,
) -> Iterable:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        max_length (int): The length of concatenated tokens
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    if is_local:
        if os.path.isdir(dataset_name):
            # only jsonl for now
            data_files = glob(f"{dataset_name}/*")
        else:
            data_files = dataset_name
        dataset = hf_datasets.load_dataset(
            "json",
            data_files=data_files,
            split=split,
            streaming=streaming,
            num_proc=num_workers if not streaming else None,
        )
    else:
        dataset = hf_datasets.load_dataset(
            path=dataset_name, name=data_subset, split=split, streaming=streaming
        )
    if shuffle:
        print("Shuffling dataset")
        dataset = dataset.shuffle(seed=seed)
    # dataset = IterableBaseDataset(name="hf_data", data=hf_dataset)
    return dataset
