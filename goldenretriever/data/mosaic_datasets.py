# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

from functools import partial
import os
from itertools import islice
from pathlib import Path
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

import numpy as np
import torch
import transformers
from composer.core.data_spec import DataSpec
from composer.core.types import Batch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from goldenretriever.common.log import get_logger

from composer.utils import dist, get_file, parse_uri
import datasets as hf_datasets

from streaming.base.world import World
from streaming.base.dataset import _Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event, Lock

from goldenretriever.common.model_inputs import ModelInputs

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
        passage_batch_size: int = 8,
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
        self.tokenizer = tokenizer
        # self.max_seq_len = max_seq_len
        self.passage_batch_size = passage_batch_size

    # How to tokenize a text sample to a token sample
    def _tokenize(self, sample: Mapping) -> Dict[str, List[int]]:
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                "If tokenizing on-the-fly, tokenizer must have a pad_token_id"
            )

        max_positives = -1
        max_negatives = -1
        max_hard_negatives = -1
        max_passages = -1
        max_question_length = 40
        max_passage_length = 40

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

        question = self.tokenizer(
            sample["question"], max_length=max_question_length, truncation=True
        )

        passage = positives + negatives + hard_negatives
        if max_passages != -1:
            passage = passage[:max_passages]

        passage = self.tokenizer(
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
        batch = []
        passages_in_batch = {}
        for sample in map(self.__getitem__, self._each_sample_id(it)):
            if len(passages_in_batch) >= self.passage_batch_size:
                # create the batch dict
                batch_dict = ModelInputs(
                    dict(
                        sample_idx=[s["id"] for s in batch],
                        questions=[s["question"] for s in batch],
                        passages=list(passages_in_batch.values()),
                        positives_pssgs=[s["positive_pssgs"] for s in batch],
                        positives=[s["positives"] for s in batch],
                    )
                )
                # split the batch if needed
                # if len(batch) > question_batch_size:
                #     for splited_batch in split_batch(batch_dict, question_batch_size):
                #         yield splited_batch
                # else:
                yield batch_dict

                # reset batch
                batch = []
                passages_in_batch = {}

            batch.append(sample)
            # yes it's a bit ugly but it works :)
            # count the number of passages in the batch and stop if we reach the limit
            # we use a set to avoid counting the same passage twice
            # we use a tuple because set doesn't support lists
            # we use input_ids as discriminator
            passages_in_batch.update(
                {tuple(passage["input_ids"]): passage for passage in sample["passage"]}
            )
        # yield from map(self.__getitem__, self._each_sample_id(it))
        wait([prepare_future, ready_future], return_when="FIRST_EXCEPTION")
        it.exit()


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

    def __call__(self, batch: Any, *args, **kwargs) -> Any:
        # convert questions and passages to a batch
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


def build_text_dataloader(
    # cfg: DictConfig,
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
) -> DataSpec:
    # assert (
    #     cfg.name == "text"
    # ), f"Tried to build text dataloader with cfg.name={cfg.name}"

    # get kwargs
    # mlm_probability = cfg.dataset.pop("mlm_probability", None)
    # eos_token_id = cfg.dataset.pop("eos_token_id", None)
    # bos_token_id = cfg.dataset.pop("bos_token_id", None)

    # streams = build_streams(cfg.dataset)

    # build dataset potentially with streams
    dataset = StreamingGoldenRetrieverDataset(
        tokenizer=tokenizer,
        streams=[Stream(local=path)],
        batch_size=device_batch_size,
        **cfg.dataset,
    )

    # collate_fn = transformers.DataCollatorForLanguageModeling(
    #     tokenizer=dataset.tokenizer,
    #     mlm=mlm_probability is not None,
    #     mlm_probability=mlm_probability,
    # )

    # if (eos_token_id is not None) or (bos_token_id is not None):
    #     # Note: Will raise an error if both are non-None
    #     collate_fn = ConcatenatedSequenceCollatorWrapper(
    #         base_collator=collate_fn,
    #         eos_token_id=eos_token_id,
    #         bos_token_id=bos_token_id,
    #     )

    # collate_fn =
    def collate_fn(batch: Any, *args, **kwargs) -> Any:
        # convert questions and passages to a batch
        questions = convert_to_batch(batch.questions)
        passages = convert_to_batch(batch.passages)

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

    detected_cpu_count = os.cpu_count() or 1
    detected_cpus_with_margin = detected_cpu_count - 8
    num_cpus_to_use = max(1, detected_cpus_with_margin)

    dl = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=8,
        drop_last=False,
        num_workers=num_cpus_to_use,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        timeout=0,
    )

    # If we pretokenized, we may not have padding, in which case the
    # tokenizer may not have a pad_token_id. In this case, we can
    # just use the default token counting function. This is correct
    # because we do not support training on pretokenized data with padding,
    # and if tokenizing on the fly, we require that the tokenizer has a pad token.
    # token_counting_func = None
    # if tokenizer.pad_token_id is not None:
    #     token_counting_func = get_tokens_per_batch_func()

    # return DataSpec(dataloader=dl, get_num_tokens_in_batch=token_counting_func)
    return dl


# class ConcatenatedSequenceCollatorWrapper:
#     """Collator wrapper to add sequence_id to batch."""

#     def __init__(
#         self,
#         base_collator: Callable,
#         eos_token_id: Optional[int] = None,
#         bos_token_id: Optional[int] = None,
#     ):
#         self.base_collator = base_collator
#         if (eos_token_id is None) and (bos_token_id is None):
#             raise ValueError(
#                 "Must supply a value for either eos_token_id or bos_token_id, but got None for both."
#             )
#         if (eos_token_id is not None) and (bos_token_id is not None):
#             raise ValueError(
#                 "Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. "
#                 + "Please supply `eos_token_id` if sequences end with an EOS token, or use "
#                 + "`bos_token_id` if sequences start with a BOS token."
#             )

#         if eos_token_id is None:
#             self.split_token_id = cast(int, bos_token_id)
#             self.bos_mode = True
#         else:
#             self.split_token_id = eos_token_id
#             self.bos_mode = False

#     def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
#         batch = self.base_collator(examples)
#         batch["sequence_id"] = self.get_sequence_id_from_batch(batch)
#         return batch

#     def get_sequence_id_from_batch(
#         self, batch: Dict[str, torch.Tensor]
#     ) -> torch.Tensor:
#         is_separator = torch.eq(batch["input_ids"], self.split_token_id)
#         cumulative_sep = torch.cumsum(is_separator, dim=1).to(batch["input_ids"].dtype)
#         # If separator token is bos, we're already done
#         if self.bos_mode:
#             return cumulative_sep

#         # If separator token is eos, right shift 1 space
#         left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
#         return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)


# # def build_streams(**kwargs: Any) -> Optional[Sequence[Stream]]:
# #     # streams_dict = dataset_cfg.pop("streams", None)
# #     # build streams
# #     streams = None
# #     if kwargs is not None and len(kwargs) > 0:
# #         streams = []
# #         for _, stream in kwargs.items():
# #             # stream is the streams kwargs
# #             # fwd all kwargs with **stream allows streaming to check args
# #             streams.append(Stream(**stream))
# #     return streams


# def get_tokens_per_batch_func(decoder_only: bool = True) -> Callable[[Batch], int]:
#     """Returns a callable that counts the number of tokens in a batch.

#     Args:
#         pad_token_id (int): The id of the padding token.
#         decoder_only (bool, optional): Whether to expect the batch to just contain ``input_ids`` (decoder only)
#             or to also contain ``decoder_input_ids`` (encoder decoder). Defaults to ``True``.

#     Returns:
#         Callable[[Batch], int]: A callable that counts the number of tokens in a batch.
#     """

#     def get_num_samples_in_batch(batch: Batch) -> int:
#         if not isinstance(batch, Mapping) or (
#             "attention_mask" not in batch and "input_ids" not in batch
#         ):
#             raise ValueError(
#                 "get_tokens_per_batch_func() requires a batch with an attention_mask key or an input_ids key"
#             )

#         if not decoder_only and "decoder_attention_mask" not in batch:
#             raise ValueError(
#                 "get_tokens_per_batch_func() for encoder decoder requires a batch with a decoder_attention_mask key"
#             )

#         # Count number of non padding tokens in batch
#         if "attention_mask" in batch:
#             input_ids_tokens = int(torch.sum(batch["attention_mask"]).item())
#         else:
#             input_ids_tokens = batch["input_ids"].numel()

#         # For encoder decoder models only
#         decoder_input_ids_tokens = 0
#         if not decoder_only:
#             decoder_input_ids_tokens = int(
#                 torch.sum(batch["decoder_attention_mask"]).item()
#             )

#         return input_ids_tokens + decoder_input_ids_tokens

#     return get_num_samples_in_batch


def build_from_hf(
    # cfg: DictConfig,
    paths: Union[str, List[str]],
    # dataset_name: str,
    # max_seq_len: int,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
    preprocessing_fn=None,
    **kwargs: Any,
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
    # dataset_name = cfg.hf_name
    # HF datasets does not support a split with dashes,so we replace split
    # dashes with underscore.
    # split = cfg.split.replace("-", "_")
    # kwargs = cfg.get("hf_kwargs", {})
    # proto_preprocessing_fn = cfg.get("preprocessing_fn")
    # if isinstance(proto_preprocessing_fn, dict) or isinstance(
    #     proto_preprocessing_fn, DictConfig
    # ):
    #     preprocessing_fn = self.get_preprocessing_fn_from_dict(proto_preprocessing_fn)
    # else:
    #     preprocessing_fn = self.get_preprocessing_fn_from_str(
    #         proto_preprocessing_fn, dataset_name
    #     )

    if isinstance(paths, str):
        paths = [
            (
                Path(paths)
                # if Path(paths).is_absolute()
                # else self.project_folder / paths
            )
        ]
    else:
        paths = [
            Path(path)  # if Path(path).is_absolute() else self.project_folder / path
            for path in paths
        ]

    # read the data and put it in a placeholder list
    for path in paths:
        if not path.exists():
            raise ValueError(f"{path} does not exist")

    signal_file_path = f".node_{dist.get_node_rank()}_local_rank0_data_prep_completed"

    # Non local rank 0 ranks will wait here for local rank 0 to finish the data processing.
    # Once local rank 0 is done, the datasets are all cached on disk, and all other ranks
    # can just read them.
    if dist.get_local_rank() != 0:
        logger.debug("Waiting for local_rank 0 to finish data prep")
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            pass

    error: Optional[Exception] = None
    tokenized_dataset = None
    try:
        dataset = hf_datasets.load_dataset(
            "json",
            data_files=[str(p) for p in paths],  # datasets needs str paths and not Path
            split=split,
            **kwargs,
        )

        # def dataset_mapper(example: Dict):
        #     if preprocessing_fn is not None:
        #         example = preprocessing_fn(example)
        #     return _tokenize_formatted_example(example, tokenizer)

        max_positives = -1
        max_negatives = -1
        max_hard_negatives = -1
        max_passages = -1
        max_question_length = 40
        max_passage_length = 40
        fn_kwargs = dict(
            tokenizer=tokenizer,
            max_positives=max_positives,
            max_negatives=max_negatives,
            max_hard_negatives=max_hard_negatives,
            max_passages=max_passages,
            max_question_length=max_question_length,
            max_passage_length=max_passage_length,
        )

        def load_fn(
            sample: Dict,
            tokenizer: PreTrainedTokenizerBase,
            max_positives: int,
            max_negatives: int,
            max_hard_negatives: int,
            max_passages: int = -1,
            max_question_length: int = 256,
            max_passage_length: int = 128,
            *args,
            **kwargs,
        ) -> Dict:
            # remove duplicates and limit the number of passages
            positives = list(set([p["text"] for p in sample["positive_ctxs"]]))
            if max_positives != -1:
                positives = positives[:max_positives]

            negatives = list(set([n["text"] for n in sample["negative_ctxs"]]))
            if max_negatives != -1:
                negatives = negatives[:max_negatives]

            hard_negatives = list(
                set([h["text"] for h in sample["hard_negative_ctxs"]])
            )
            if max_hard_negatives != -1:
                hard_negatives = hard_negatives[:max_hard_negatives]

            question = tokenizer(
                sample["question"], max_length=max_question_length, truncation=True
            )

            passage = positives + negatives + hard_negatives
            if max_passages != -1:
                passage = passage[:max_passages]

            passage = tokenizer(passage, max_length=max_passage_length, truncation=True)

            # invert the passage data structure from a dict of lists to a list of dicts
            passage = [dict(zip(passage, t)) for t in zip(*passage.values())]

            output = dict(
                question=question,
                passage=passage,
                positive_pssgs=passage[: len(positives)],
                positives=positives,
                negatives=negatives,
                hard_negatives=hard_negatives,
            )
            return output

        detected_cpu_count = os.cpu_count() or 1
        detected_cpus_with_margin = detected_cpu_count - 8
        num_cpus_to_use = max(1, detected_cpus_with_margin)
        map_kwargs = dict(
            function=load_fn,
            fn_kwargs=fn_kwargs,
            batched=False,
            # remove_columns=columns_to_remove,
            num_proc=num_cpus_to_use,
            desc="Tokenizing dataset",
        )

        # columns_to_remove = list(dataset[0].keys())
        tokenized_dataset = dataset.map(**map_kwargs)

        # pad_token_id = tokenizer.pad_token_id

        # def filter_long_or_empty_examples(example: Dict) -> bool:
        #     less_than_max_seq_len = len(example["input_ids"]) < max_seq_len
        #     non_empty_input = len(example["input_ids"]) > 0
        #     non_empty_labels = len(example["labels"]) > 0
        #     non_padding_response = any(
        #         token_id != pad_token_id for token_id in example["labels"]
        #     )
        #     return (
        #         less_than_max_seq_len
        #         and non_empty_input
        #         and non_empty_labels
        #         and non_padding_response
        #     )

        # filtered_dataset = tokenized_dataset.filter(
        #     filter_long_or_empty_examples,
        #     num_proc=num_cpus_to_use,
        #     desc="Filtering out long prompts",
        # )

        # examples_removed = len(tokenized_dataset) - len(filtered_dataset)
        # if examples_removed > 0:
        #     logger.warn(
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
    assert tokenized_dataset is not None
    return tokenized_dataset


# def build_from_streaming(self, *args: Any, **kwargs: Any) -> StreamingFinetuningDataset:
#     return StreamingFinetuningDataset(*args, **kwargs)

# from torch.utils.data.distributed import DistributedSampler

# class GoldenRetrieverDistributedSampler(DistributedSampler):
#     def __iter__(self) -> os.Iterator:
#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
#         else:
#             indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices)
#             if padding_size <= len(indices):
#                 indices += indices[:padding_size]
#             else:
#                 indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices = indices[:self.total_size]
#         assert len(indices) == self.total_size

#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples

#         return iter(indices)


# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == "__main__":
    import argparse

    from llmfoundry.utils.builders import build_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neox-20b",
        help="the name of the tokenizer to use",
    )
    parser.add_argument(
        "--local_path",
        type=str,
        required=True,
        help="the path to the local copy of the dataset",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        default=None,
        help="the path to the remote copy to stream from (optional)",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="which split of the dataset to use"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=32, help="max sequence length to test"
    )

    args = parser.parse_args()

    if args.remote_path is not None:
        print(
            f"Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}"
        )
    else:
        print(f"Reading {args.split} split from {args.local_path}")

    cfg = {
        "name": "text",
        "dataset": {
            "local": args.local_path,
            "remote": args.remote_path,
            "split": args.split,
            "shuffle": False,
            "max_seq_len": args.max_seq_len,
            "keep_zip": True,  # in case we need compressed files after testing
        },
        "drop_last": False,
        "num_workers": 4,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    tokenizer_name = args.tokenizer
    tokenizer_kwargs = {"model_max_length": args.max_seq_len}
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    loader = build_text_dataloader(cfg, tokenizer, device_batch_size).dataloader
    assert isinstance(loader, DataLoader)
    assert isinstance(loader.dataset, StreamingTextDataset)
    tokenizer = loader.dataset.tokenizer

    for batch_ix, batch in enumerate(islice(loader, 5)):
        print("\n")
        print("#" * 20, f"Batch {batch_ix}", "#" * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch["input_ids"]):
            print("-" * 20, f" Sample {sample_ix} ", "-" * 20)
            print(tokenizer.decode(token_sample))
