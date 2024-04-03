# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

from glob import glob
import os
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from itertools import islice
from pathlib import Path
from threading import Event, Lock
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

import datasets as hf_datasets
import numpy as np
import torch
import transformers

# from composer.core.data_spec import DataSpec
# from composer.core.types import Batch
# from composer.utils import dist, get_file, parse_uri
# from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Stream, StreamingDataLoader, StreamingDataset
from streaming.base.dataset import _Iterator
from streaming.base.world import World
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from goldenretriever.common.log import get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.utils import HardNegativesManager

logger = get_logger(__name__)


class StreamingGoldenRetrieverDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
    """

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        # max_seq_len: int,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[Union[int, str]] = None,
        predownload: Optional[int] = None,
        cache_limit: Optional[Union[int, str]] = None,
        partition_algo: str = "relaxed",
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = "py1e",
        shuffle_seed: int = 9176,
        shuffle_block_size: Optional[int] = None,
        sampling_method: str = "balanced",
        sampling_granularity: int = 1,
        batching_method: str = "random",
        # golden retriever specific
        question_batch_size: int = 32,
        passage_batch_size: int = 32,
        max_positives: int = -1,
        max_negatives: int = -1,
        max_hard_negatives: int = -1,
        max_passages: int = -1,
        max_question_length: int = 40,
        max_passage_length: int = 40,
        **kwargs: Any,
    ):

        if len(kwargs) > 0:
            raise ValueError(
                f"StreamingTextDataset() got an unexpected keyword argument: {kwargs}"
            )

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f"local directory {local} does not contain split {split}"
                    )

        # TODO: discover where yamls are being converted incorrect, but temporary workaround
        if isinstance(shuffle_block_size, float):
            shuffle_block_size = int(shuffle_block_size)

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            sampling_method=sampling_method,
            sampling_granularity=sampling_granularity,
            batching_method=batching_method,
        )
        self.name = name
        self.tokenizer = tokenizer
        self.question_batch_size = question_batch_size
        self.passage_batch_size = passage_batch_size
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_passages = max_passages
        self.max_question_length = max_question_length
        self.max_passage_length = max_passage_length

        # self.hn_manager = HardNegativesManager(tokenizer, max_length=max_passage_length)

    # How to tokenize a text sample to a token sample
    def _tokenize(self, sample: Mapping) -> Dict[str, List[int]]:
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                "If tokenizing on-the-fly, tokenizer must have a pad_token_id"
            )

        # remove duplicates and limit the number of passages
        positives = list(set([p["text"] for p in sample["positive_ctxs"]]))
        if self.max_positives != -1:
            positives = positives[: self.max_positives]

        negatives = list(set([n["text"] for n in sample["negative_ctxs"]]))
        if self.max_negatives != -1:
            negatives = negatives[: self.max_negatives]

        hard_negatives = list(set([h["text"] for h in sample["hard_negative_ctxs"]]))
        if self.max_hard_negatives != -1:
            hard_negatives = hard_negatives[: self.max_hard_negatives]

        question = self.tokenizer(
            sample["question"], max_length=self.max_question_length, truncation=True
        )

        passage = positives + negatives + hard_negatives
        if self.max_passages != -1:
            passage = passage[: self.max_passages]

        passage = self.tokenizer(
            passage, max_length=self.max_passage_length, truncation=True
        )

        # invert the passage data structure from a dict of lists to a list of dicts
        passage = [dict(zip(passage, t)) for t in zip(*passage.values())]

        output = dict(
            id=sample["id"],
            question=question,
            passage=passage,
            positive_pssgs=passage[: len(positives)],
            positives=positives,
            negatives=negatives,
            hard_negatives=hard_negatives,
        )
        return output

    def _read_binary_tokenized_sample(self, sample: Dict[str, Any]) -> torch.Tensor:
        return torch.from_numpy(
            np.frombuffer(sample["tokens"], dtype=np.int64)[: self.max_seq_len].copy()
        )

    # How to process a sample
    def __getitem__(self, idx: int) -> Union[Dict[str, List[int]], torch.Tensor]:
        sample = super().__getitem__(idx)
        # if "text" in sample:
        token_sample = self._tokenize(sample)
        # elif "tokens" in sample:
        #     token_sample = self._read_binary_tokenized_sample(sample)
        # else:
        #     raise RuntimeError(
        #         "StreamingTextDataset needs samples to have a `text` or `tokens` column"
        #     )
        return token_sample

    # def __len__(self) -> int:
    #     """Get the length as a PyTorch IterableDataset.

    #     Returns:
    #         int: Dataset length.
    #     """
    #     # raise NotImplementedError("StreamingDataset does not support __len__")
    #     return None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        # Exit the threads that are pre-downloading and iterating the shards for previous epoch, if
        # it exists.
        if hasattr(self, "_iterator"):
            self._iterator.exit()

        # For exception handling.
        if not hasattr(self, "_executor"):
            self._executor = ThreadPoolExecutor()
        if not hasattr(self, "_event"):
            self._event = Event()
        elif self._event.is_set():
            raise RuntimeError("Background thread failed. Check other traceback.")

        # Discover where we left off, if there is a checkpoint, or start at the next epoch.
        # Also pre-increment the epoch counter.
        world = World()
        epoch, sample_in_epoch = self._resume_incr_epoch(world)

        # Get this worker's partition of samples to process.
        sample_ids = self._get_work(world, epoch, sample_in_epoch)
        if not len(sample_ids):  # Resumed at end of epoch, out of samples.
            return

        # Iterate over the samples while downloading ahead.
        self._iterator = it = _Iterator(sample_ids)
        prepare_future = self._executor.submit(self._prepare_thread, it)
        prepare_future.add_done_callback(self.on_exception)
        ready_future = self._executor.submit(self._ready_thread, it)
        ready_future.add_done_callback(self.on_exception)
        # Iterate over the samples and accumulate passage_batch_size samples at a time
        # batch = []
        # passages_in_batch = {}
        # for sample in map(self.__getitem__, self._each_sample_id(it)):
        #     if len(passages_in_batch) >= self.passage_batch_size:
        #         # create the batch dict
        #         batch_dict = ModelInputs(
        #             dict(
        #                 sample_idx=[s["id"] for s in batch],
        #                 questions=[s["question"] for s in batch],
        #                 passages=list(passages_in_batch.values()),
        #                 positives_pssgs=[s["positive_pssgs"] for s in batch],
        #                 positives=[s["positives"] for s in batch],
        #             )
        #         )
        #         # split the batch if needed
        #         if len(batch) > self.question_batch_size:
        #             for splited_batch in self.split_batch(
        #                 batch_dict, self.question_batch_size
        #             ):
        #                 yield splited_batch
        #         else:
        #             yield batch_dict

        #         # reset batch
        #         batch = []
        #         passages_in_batch = {}

        #     batch.append(sample)
        #     # yes it's a bit ugly but it works :)
        #     # count the number of passages in the batch and stop if we reach the limit
        #     # we use a set to avoid counting the same passage twice
        #     # we use a tuple because set doesn't support lists
        #     # we use input_ids as discriminator
        #     passages_in_batch.update(
        #         {tuple(passage["input_ids"]): passage for passage in sample["passage"]}
        #     )
        #     # check for hard negatives and add with a probability of 0.1
        #     if self.hn_manager is not None:
        #         if sample["id"] in self.hn_manager:
        #             passages_in_batch.update(
        #                 {
        #                     tuple(passage["input_ids"]): passage
        #                     for passage in self.hn_manager.get(sample["id"])
        #                 }
        #             )
        #         else:
        #             print(f"Sample {sample['id']} not in hn_manager")
        yield from map(self.__getitem__, self._each_sample_id(it))
        wait([prepare_future, ready_future], return_when="FIRST_EXCEPTION")
        it.exit()

    @staticmethod
    def get_num_samples_in_batch(batch: Dict) -> int:
        """
        Get the number of samples in a batch.

        Args:
            batch (Dict): A batch of data.

        Returns:
            int: The number of samples in the batch.
        """
        try:
            return batch["questions"]["input_ids"].shape[0]
        except KeyError:
            raise ValueError("Batch must contain `questions` key.")

    @staticmethod
    def get_num_tokens_in_batch(batch: Dict) -> int:
        """
        Get the number of tokens in a batch.

        Args:
            batch (Dict): A batch of data.

        Returns:
            int: The number of tokens in the batch.
        """
        try:
            return (
                batch["questions"]["input_ids"].shape[1]
                + batch["passages"]["input_ids"].shape[1]
            )
        except KeyError:
            raise ValueError("Batch must contain `questions` and `passages` keys.")

    @staticmethod
    def split_batch(
        batch: Union[Dict[str, Any], ModelInputs], microbatch_size: int
    ) -> List[ModelInputs]:
        """
        Split a batch into multiple batches of size `question_batch_size` while keeping
        the same number of passages.
        """

        # def split_fn(x):
        #     return [
        #         x[i : i + microbatch_size] for i in range(0, len(x), microbatch_size)
        #     ]

        # # split the sample_idx
        # sample_idx = split_fn(batch["sample_idx"])
        # # split the questions
        # questions = split_fn(batch["questions"])
        # # split the positives
        # positives = split_fn(batch["positives"])
        # # split the positives_pssgs
        # positives_pssgs = split_fn(batch["positives_pssgs"])

        # # collect the new batches
        # batches = []
        # for i in range(len(questions)):
        #     batches.append(
        #         ModelInputs(
        #             dict(
        #                 sample_idx=sample_idx[i],
        #                 questions=questions[i],
        #                 passages=batch["passages"],
        #                 positives=positives[i],
        #                 positives_pssgs=positives_pssgs[i],
        #             )
        #         )
        #     )
        # return batches

        if microbatch_size is None:
            return [batch]

        def split_fn(x):
            if isinstance(x, list):
                return [
                    x[i : i + microbatch_size]
                    for i in range(0, len(x), microbatch_size)
                ]
            elif isinstance(x, torch.Tensor):
                return torch.split(x, microbatch_size, dim=0)
            elif isinstance(x, dict):
                # split the dict values into microbatches while
                # keeping the keys the same
                return [
                    dict((k, v[i : i + microbatch_size]) for k, v in x.items())
                    for i in range(0, len(x[list(x.keys())[0]]), microbatch_size)
                ]
            else:
                raise ValueError(f"Unsupported type {type(x)}")

        # split the sample_idx
        sample_idx = split_fn(batch["sample_idx"])
        # split the questions
        questions = split_fn(batch["questions"])
        # split the labels
        labels = split_fn(batch["labels"])
        # split the positives
        positives = split_fn(batch["positives"])

        # collect the new batches
        batches = []
        for i in range(len(questions)):
            batches.append(
                ModelInputs(
                    dict(
                        sample_idx=sample_idx[i],
                        questions=questions[i],
                        passages=batch["passages"],
                        positives=positives[i],
                        labels=labels[i],
                    )
                )
            )
        return batches


class GoldenRetrieverCollator:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.padding_ops = {
            "input_ids": partial(
                self.pad_sequence,
                value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": partial(self.pad_sequence, value=0),
            "token_type_ids": partial(
                self.pad_sequence,
                value=self.tokenizer.pad_token_type_id,
            ),
        }

    @staticmethod
    def pad_sequence(
        sequence: Union[List, torch.Tensor],
        length: int,
        value: Any = None,
        pad_to_left: bool = False,
    ) -> Union[List, torch.Tensor]:
        """
        Pad the input to the specified length with the given value.

        Args:
            sequence (:obj:`List`, :obj:`torch.Tensor`):
                Element to pad, it can be either a :obj:`List` or a :obj:`torch.Tensor`.
            length (:obj:`int`, :obj:`str`, optional, defaults to :obj:`subtoken`):
                Length after pad.
            value (:obj:`Any`, optional):
                Value to use as padding.
            pad_to_left (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True`, pads to the left, right otherwise.

        Returns:
            :obj:`List`, :obj:`torch.Tensor`: The padded sequence.

        """
        padding = [value] * abs(length - len(sequence))
        if isinstance(sequence, torch.Tensor):
            if len(sequence.shape) > 1:
                raise ValueError(
                    f"Sequence tensor must be 1D. Current shape is `{len(sequence.shape)}`"
                )
            padding = torch.as_tensor(padding)
        if pad_to_left:
            if isinstance(sequence, torch.Tensor):
                return torch.cat((padding, sequence), -1)
            return padding + sequence
        if isinstance(sequence, torch.Tensor):
            return torch.cat((sequence, padding), -1)
        return sequence + padding

    def convert_to_batch(
        self, samples: Any, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Convert the list of samples to a batch.

        Args:
            samples (:obj:`List`):
                List of samples to convert to a batch.

        Returns:
            :obj:`Dict[str, torch.Tensor]`: The batch.
        """
        # invert questions from list of dict to dict of list
        samples = {k: [d[k] for d in samples] for k in samples[0]}
        # get max length of questions
        max_len = max(len(x) for x in samples["input_ids"])
        # pad the questions
        for key in samples:
            if key in self.padding_ops:
                samples[key] = torch.as_tensor(
                    [self.padding_ops[key](b, max_len) for b in samples[key]]
                )
        return samples

    # def hard_negatives_augmentation(self, batch: Any, *args, **kwargs) -> Any:
    #     for sample in batch:
    #         if sample["id"] in self.hn_manager:
    #             batch["passages"].extend(self.hn_manager.get(sample["id"]))
    #             # passages_in_batch.update(
    #             #     {
    #             #         tuple(passage["input_ids"]): passage
    #             #         for passage in self.hn_manager.get(sample["id"])
    #             #     }
    #             # )

    def __call__(self, batch: Any, *args, **kwargs) -> Any:
        # convert questions and passages to a batch
        # batch = ModelInputs(batch)

        # for sample in batch:
        # if len(passages_in_batch) >= self.passage_batch_size:
        # create the batch dict
        # self.hard_negatives_augmentation(batch)

        passages_in_batch = {}
        for sample in batch:
            passages_in_batch.update(
                {tuple(passage["input_ids"]): passage for passage in sample["passage"]}
            )
        batch = ModelInputs(
            dict(
                sample_idx=[s["id"] for s in batch],
                questions=[s["question"] for s in batch],
                passages=list(passages_in_batch.values()),
                positives_pssgs=[s["positive_pssgs"] for s in batch],
                positives=[s["positives"] for s in batch],
            )
        )
        # # split the batch if needed
        # if len(batch) > self.question_batch_size:
        #     for splited_batch in self.split_batch(
        #         batch_dict, self.question_batch_size
        #     ):
        #         yield splited_batch
        # else:
        #     yield batch_dict

        # reset batch
        # batch = []
        # passages_in_batch = {}

        # batch.append(sample)
        # yes it's a bit ugly but it works :)
        # count the number of passages in the batch and stop if we reach the limit
        # we use a set to avoid counting the same passage twice
        # we use a tuple because set doesn't support lists
        # we use input_ids as discriminator

        questions = self.convert_to_batch(batch.questions)
        passages = self.convert_to_batch(batch.passages)

        # build an index to map the position of the passage in the batch
        passage_index = {tuple(c["input_ids"]): i for i, c in enumerate(batch.passages)}

        # now we can create the labels
        labels = torch.zeros(
            questions["input_ids"].shape[0], passages["input_ids"].shape[0]
        )
        # iterate over the questions and set the labels to 1 if the passage is positive
        for sample_idx in range(len(questions["input_ids"])):
            for pssg in batch["positives_pssgs"][sample_idx]:
                # get the index of the positive passage
                index = passage_index[tuple(pssg["input_ids"])]
                # set the label to 1
                labels[sample_idx, index] = 1

        model_inputs = ModelInputs(
            {
                "questions": questions,
                "passages": passages,
                "labels": labels,
                "positives": batch["positives"],
                "sample_idx": batch["sample_idx"],
            }
        )
        return model_inputs


class GoldenStreamingDataLoader(StreamingDataLoader):
    """A streaming data loader.

    Provides an additional checkpoint/resumption interface, for which it tracks the number of
    samples seen by the model this rank.

    Args:
        *args: List arguments.
        **kwargs: Keyword arguments.
    """

    def __init__(self, *args, **kwargs) -> None:  # pyright: ignore
        super().__init__(*args, **kwargs)
        self.num_samples_yielded = 0

    def _get_batch_size(self, batch: Any) -> int:
        """Get the number of samples in a batch.

        Args:
            batch (Any): The batch.

        Returns:
            int: Number of samples.
        """
        # if isinstance(batch, (dict, BatchEncoding, BatchFeature)):
        #     for value in batch.values():
        #         return len(value)
        #     raise ValueError('Batch is empty')
        # elif isinstance(batch, Tensor):
        #     return len(batch)
        # else:
        #     return len(batch[0])
        return batch["questions"]["input_ids"].get_size(0)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over this DataLoader, yielding batches.

        Also tracks the number of samples seen this rank.

        Returns:
            Iterator[Any]: Each batch.
        """
        self.num_samples_yielded = 0
        batch = []
        for batch in super().__iter__():
            self.num_samples_yielded += self._get_batch_size(batch)
            yield batch

    def state_dict(self) -> Optional[Dict[str, Any]]:
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Args:
            samples_in_epoch (int): The number of samples processed so far in the current epoch.

        Returns:
            Optional[Dict[str, Any]]: The state, if a streaming dataset.
        """
        if isinstance(self.dataset, StreamingDataset):
            world = World()
            num_samples = self.num_samples_yielded * world.num_ranks
            return self.dataset.state_dict(num_samples, False)
        return None

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        Args:
            obj (Dict[str, Any]): The state.
        """
        if isinstance(self.dataset, StreamingDataset):
            self.dataset.load_state_dict(obj)

    def __del__(self) -> None:
        """Terminate the workers during cleanup."""
        if self._iterator is not None:
            self._iterator._shutdown_workers()  # type: ignore [reportGeneralTypeIssues]


# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
# if __name__ == "__main__":
#     import argparse

#     from llmfoundry.utils.builders import build_tokenizer

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--tokenizer",
#         type=str,
#         default="EleutherAI/gpt-neox-20b",
#         help="the name of the tokenizer to use",
#     )
#     parser.add_argument(
#         "--local_path",
#         type=str,
#         required=True,
#         help="the path to the local copy of the dataset",
#     )
#     parser.add_argument(
#         "--remote_path",
#         type=str,
#         default=None,
#         help="the path to the remote copy to stream from (optional)",
#     )
#     parser.add_argument(
#         "--split", type=str, default="val", help="which split of the dataset to use"
#     )
#     parser.add_argument(
#         "--max_seq_len", type=int, default=32, help="max sequence length to test"
#     )

#     args = parser.parse_args()

#     if args.remote_path is not None:
#         print(
#             f"Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}"
#         )
#     else:
#         print(f"Reading {args.split} split from {args.local_path}")

#     cfg = {
#         "name": "text",
#         "dataset": {
#             "local": args.local_path,
#             "remote": args.remote_path,
#             "split": args.split,
#             "shuffle": False,
#             "max_seq_len": args.max_seq_len,
#             "keep_zip": True,  # in case we need compressed files after testing
#         },
#         "drop_last": False,
#         "num_workers": 4,
#     }
#     cfg = om.create(cfg)
#     device_batch_size = 2

#     tokenizer_name = args.tokenizer
#     tokenizer_kwargs = {"model_max_length": args.max_seq_len}
#     tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

#     loader = build_text_dataloader(cfg, tokenizer, device_batch_size).dataloader
#     assert isinstance(loader, DataLoader)
#     assert isinstance(loader.dataset, StreamingTextDataset)
#     tokenizer = loader.dataset.tokenizer

#     for batch_ix, batch in enumerate(islice(loader, 5)):
#         print("\n")
#         print("#" * 20, f"Batch {batch_ix}", "#" * 20)
#         for k, v in batch.items():
#             print(k, v.shape, v.dtype)
#         for sample_ix, token_sample in enumerate(batch["input_ids"]):
#             print("-" * 20, f" Sample {sample_ix} ", "-" * 20)
#             print(tokenizer.decode(token_sample))

import importlib
import logging
import os
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import datasets as hf_datasets
import huggingface_hub as hf_hub
import numpy as np
from composer.utils import dist


DOWNLOADED_FT_DATASETS_DIRPATH = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "datasets"
)
SUPPORTED_EXTENSIONS = [".csv", ".jsonl", ".parquet"]


def _is_empty_or_nonexistent(dirpath: str) -> bool:
    """Check if a directory is empty or non-existent.

    Args:
        dirpath (str): Directory path to check.

    Returns
        True if directory is empty or non-existent. False otherwise.
    """
    return not os.path.isdir(dirpath) or len(os.listdir(dirpath)) == 0


def tokenize(
    sample: Mapping,
    tokenizer,
    max_positives,
    max_negatives,
    max_hard_negatives,
    max_question_length,
    max_passages,
    max_passage_length,
) -> Dict[str, List[int]]:
    # remove duplicates and limit the number of passages
    positives = list(set([p["text"] for p in sample["positive_ctxs"]]))
    if max_positives != -1:
        positives = positives[:max_positives]

    negatives = list(set([n["text"] for n in sample["negative_ctxs"]]))
    if max_negatives != -1:
        negatives = negatives[:max_negatives]

    hard_negatives = list(set([h["text"] for h in sample["hard_negative_ctxs"]]))
    if max_hard_negatives != -1:
        hard_negatives = hard_negatives[:max_hard_negatives]

    question = tokenizer(
        sample["question"], max_length=max_question_length, truncation=True
    )

    passage = positives + negatives + hard_negatives
    if max_passages != -1:
        passage = passage[: max_passages]

    passage = tokenizer(
        passage, max_length=max_passage_length, truncation=True
    )

    # invert the passage data structure from a dict of lists to a list of dicts
    passage = [dict(zip(passage, t)) for t in zip(*passage.values())]

    output = dict(
        id=sample["id"],
        question=question,
        passage=passage,
        positive_pssgs=passage[: len(positives)],
        positives=positives,
        negatives=negatives,
        hard_negatives=hard_negatives,
    )
    return output


def build_from_hf(
    dataset_name: str,
    split: Optional[str],
    safe_load: bool,
    tokenizer: PreTrainedTokenizerBase,
    hf_kwargs: Dict[str, Any],
) -> Union[
    hf_datasets.DatasetDict,
    hf_datasets.Dataset,
    hf_datasets.IterableDatasetDict,
    hf_datasets.IterableDataset,
]:
    """Load a HuggingFace Datasets, preprocess, and tokenize.

    Note: This function will drop examples where the prompt is longer than the max_seq_len

    Args:
        cfg (DictConfig): The dataset configuration.
        max_seq_len (int): The maximum sequence length. Examples with prompts longer than this will be dropped.
        tokenizer (Tokenizer): The tokenizer to be used for tokenizing the dataset.

    Returns:
        Dataset: The tokenized dataset.
    """
    signal_file_path = f".node_{dist.get_node_rank()}_local_rank0_data_prep_completed"

    # Non local rank 0 ranks will wait here for local rank 0 to finish the data processing.
    # Once local rank 0 is done, the datasets are all cached on disk, and all other ranks
    # can just read them.
    if dist.get_local_rank() != 0:
        logger.debug("Waiting for local_rank 0 to finish data prep")
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            pass

    # hf_tokenization_logger = logging.getLogger("transformers.tokenization_utils_base")
    # sequence_length_warning_filter = SpecificWarningFilter(
    #     "Token indices sequence length is longer than the specified maximum sequence length"
    # )

    # We will trim examples later in the collate_fn, so we want to silence this warning from Hugging Face
    # hf_tokenization_logger.addFilter(sequence_length_warning_filter)

    error: Optional[Exception] = None

    detected_cpu_count = os.cpu_count() or 1
    detected_cpus_with_margin = detected_cpu_count - 8
    num_cpus_to_use = max(1, detected_cpus_with_margin)
    # filtered_dataset = None
    try:
        # if safe_load:
        #     if not os.path.isdir(dataset_name):
        #         # dataset_name is not a local dir path, download if needed.
        #         local_dataset_dir = os.path.join(
        #             DOWNLOADED_FT_DATASETS_DIRPATH, dataset_name
        #         )

        #         if _is_empty_or_nonexistent(dirpath=local_dataset_dir):
        #             # Safely load a dataset from HF Hub with restricted file types.
        #             hf_hub.snapshot_download(
        #                 dataset_name,
        #                 repo_type="dataset",
        #                 allow_patterns=["*" + ext for ext in SUPPORTED_EXTENSIONS],
        #                 token=hf_kwargs.get("token", None),
        #                 revision=hf_kwargs.get("revision", None),
        #                 local_dir_use_symlinks=False,
        #                 local_dir=local_dataset_dir,
        #             )
        #             if _is_empty_or_nonexistent(dirpath=local_dataset_dir):
        #                 raise FileNotFoundError(
        #                     f"safe_load is set to True. No data files with safe extensions {SUPPORTED_EXTENSIONS} "
        #                     + f"found for dataset {dataset_name}. "
        #                 )
        #         # Set dataset_name to the downloaded location.
        #         dataset_name = local_dataset_dir

        #     # dataset_name is a local dir path. Use the abspath to prevent confusion.
        #     dataset_name = os.path.abspath(dataset_name)

        #     # Ensure that the local dir contains only allowed file types.
        #     dataset_files = [f for _, _, files in os.walk(dataset_name) for f in files]
        #     if not all(Path(f).suffix in SUPPORTED_EXTENSIONS for f in dataset_files):
        #         raise ValueError(
        #             f"Dataset at local path {dataset_name} contains invalid file types. "
        #             + f"Allowed file types are: {SUPPORTED_EXTENSIONS}"
        #         )
        # dataset = hf_datasets.load_dataset(dataset_name, split=split, **hf_kwargs)
        if os.path.isdir(dataset_name):
            # only jsonl for now
            data_files = glob(f'{dataset_name}/*.jsonl')
        else:
            data_files = dataset_name
        dataset = hf_datasets.load_dataset(
            "json",
            data_files=data_files,
            split=split,
            # streaming=streaming,
            num_proc=num_cpus_to_use,
        )

        # def dataset_mapper(example: Dict):
        #     if preprocessing_fn is not None:
        #         example = preprocessing_fn(example)
        #     return tokenize_formatted_example(example, tokenizer)

        # columns_to_remove = list(dataset[0].keys())
        tokenized_dataset = dataset.map(
            partial(
                tokenize,
                tokenizer=tokenizer,
                max_positives=-1,
                max_negatives=-1,
                max_hard_negatives=-1,
                max_passages=-1,
                max_question_length=40,
                max_passage_length=40,
            ),
            batched=False,
            # remove_columns=columns_to_remove,
            num_proc=num_cpus_to_use,
            desc="Tokenizing dataset",
        )

        # filtered_dataset = tokenized_dataset.filter(
        #     partial(
        #         is_valid_ift_example,
        #         max_seq_len,
        #         target_prompts,
        #         target_responses,
        #         decoder_only_format,
        #     ),
        #     num_proc=num_cpus_to_use,
        #     desc="Filtering out long prompts",
        # )

        # examples_removed = len(tokenized_dataset) - len(filtered_dataset)
        # if examples_removed > 0:
        #     warnings.warn(
        #         f"Dropped {examples_removed} examples where the prompt was longer than {max_seq_len}, "
        #         + "the prompt or response was empty, or the response was all padding tokens."
        #     )
    except Exception as e:
        error = e
    # Now local rank 0 indicates to the other ranks that it is done
    if dist.get_local_rank() == 0:
        logger.debug("Local rank 0 finished data prep")
        with open(signal_file_path, "wb") as f:
            f.write(b"local_rank0_completed_data_prep")

    # All ranks sync up at this barrier, having completed data processing
    dist.barrier()

    # Last, local rank 0 cleans up the signal file
    if dist.get_local_rank() == 0:
        os.remove(signal_file_path)

    if error is not None:
        logger.error("Error during data prep")
        raise error
    logger.debug("All ranks finished data prep")

    # hf_tokenization_logger.removeFilter(sequence_length_warning_filter)

    assert tokenized_dataset is not None
    return tokenized_dataset

    # dl = DataLoader(
    #     dataset,
    #     collate_fn=collate_fn,
    #     batch_size=dataloader_batch_size,
    #     drop_last=cfg.drop_last,
    #     sampler=sampler,
    #     num_workers=cfg.num_workers,
    #     pin_memory=cfg.get('pin_memory', True),
    #     prefetch_factor=cfg.get('prefetch_factor', 2),
    #     persistent_workers=cfg.get('persistent_workers', True),
    #     timeout=cfg.get('timeout', 0),
    # )

    # token_counting_func = get_tokens_per_batch_func()

    # return DataSpec(dataloader=dl, get_num_tokens_in_batch=token_counting_func)

