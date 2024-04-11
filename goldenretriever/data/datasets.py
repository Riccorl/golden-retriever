import base64
import json
import os
from functools import partial
from pathlib import Path
import platform
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import psutil
import torch
from streaming import MDSWriter, Stream, StreamingDataset
from streaming.base.format import get_index_basename
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from goldenretriever.common.hf_utils import build_hf_dataset
from goldenretriever.common.log import get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.common.torch_utils import build_dataloader
from goldenretriever.common.utils import (
    GOLDENRETRIEVER_CACHE_DIR,
    file_exists,
    url_to_filename,
)

logger = get_logger(__name__)


class GoldenRetrieverStreamingDataset(StreamingDataset):
    """A mid-epoch-resumable streaming/caching pytorch IterableDataset.

    Features elastically deterministic shuffling, which enables fast mid-epoch resumption.

    Checkpoints are represented in JSON as follows:

    .. code-block:: json

        {
            "epoch" :"int",
            "sample_in_epoch": "int",
            "shuffle_seed": "int",
            "num_canonical_nodes": "int"
        }

    StreamingDataset init takes two kinds of arguments:

    * What to iterate:

      * One or more streams (you must provide either ``streams`` or ``remote``/``local``):

        * ``streams``
        * ``remote``
        * ``local``

      * Knobs to control streaming behavior, which, if multiple streams are provided,
        become defaults applied to each of them:

        * ``split``
        * ``download_retry``
        * ``download_timeout``
        * ``validate_hash``
        * ``keep_zip``

      * Absolute dataset size, if streams were weighted relatively:

        * ``epoch_size``

    * How to iterate:

      * Shard lifecycle:

        * ``predownload``
        * ``cache_limit``

      * Sampling:

        * ``sampling_method``
        * ``sampling_granularity``

      * Determinism:

        * ``partition_algo``
        * ``num_canonical_nodes``
        * ``batch_size``

      * Shuffling:

        * ``shuffle``
        * ``shuffle_algo``
        * ``shuffle_seed``
        * ``shuffle_block_size``

      * Batching:

        * ``batching_method``


    Args:
        streams (Sequence[Stream], optional): One or more streams to stream/cache samples from,
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
            ``False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced
            across all streams. If ``None``, takes its value from the total number of underlying
            samples. Provide this field if you are weighting streams relatively to target a larger
            or smaller epoch size. Defaults to ``None``. Can also take in human-readable number
            abbreviations (e.g., ``"100k"``, ``"64M"``, ``"77b"``, etc). Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[int, str], optional): Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s)
            may be evicted (deleted from the local cache) in order to stay under the limit.
            Set to ``None`` to disable shard eviction. Supports integer bytes as well as string
            human-readable bytes (e.g., ``100b``, ``64kb``, ``77mb``, and so on). Defaults to
            ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``relaxed``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. The sample space is divided evenly according to the number of canonical
            nodes. The higher the value, the more independent non-overlapping paths the
            StreamingDataset replicas take through the shards per model replica (increasing data
            source diversity). If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.

            .. note::

                For sequential sample ordering, set ``shuffle`` to ``False`` and
                ``num_canonical_nodes`` to 1.
        batch_size (int, optional): Per-device batch size, the same as what is passed to the
            DataLoader. This affects how the dataset is partitioned over the workers and is
            necessary for deterministic resumption and optimal performance. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
        replication (int, optional): Determines how many consecutive devices will receive the same
            samples. Useful for training with tensor or sequence parallelism, where multiple
            devices need to see the same partition of the dataset. Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
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
        sampling_method: str = "balanced",
        sampling_granularity: int = 1,
        partition_algo: str = "relaxed",
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = "py1e",
        shuffle_seed: int = 9176,
        shuffle_block_size: Optional[int] = None,
        batching_method: str = "random",
        allow_unsafe_types: bool = True,
        replication: Optional[int] = None,
        # golden retriever specific
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
                f"GoldenRetrieverStreamingDataset() got an unexpected keyword argument: {kwargs}"
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

        # we initialize subclass specific attributes first
        # because we need to use them in case of preprocessing
        self.name = name
        self.tokenizer = tokenizer
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_passages = max_passages
        self.max_question_length = max_question_length
        self.max_passage_length = max_passage_length

        local = self._preprocess_to_md(
            local, self._tokenize
        )  # , tokenizer=tokenizer, preprocess=True)

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
            allow_unsafe_types=allow_unsafe_types,
            replication=replication,
        )

    def __getitem__(self, idx: int) -> Union[Dict[str, List[int]], torch.Tensor]:
        sample = super().__getitem__(idx)

        # check if the sample has been tokenized already
        if isinstance(sample["question"], str):
            return self._tokenize(sample)
        else:
            # TODO: check how to handle this
            return sample

    def _read_binary_tokenized_sample(self, sample: Dict[str, Any]) -> torch.Tensor:
        return {
            k: (
                torch.from_numpy(np.frombuffer(v, dtype=np.int64))
                if isinstance(v, np.ndarray)
                else v
            )
            for k, v in sample.items()
        }

        # torch.from_numpy(
        #     np.frombuffer(sample["tokens"], dtype=np.int64)[: self.max_seq_len].copy()
        # )

    # How to tokenize a text sample to a token sample
    def _tokenize(self, sample: Mapping) -> Dict[str, List[int]]:
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

        text_pair = sample.get("doc_topic", None)
        question = self.tokenizer(
            sample["question"],
            text_pair=text_pair,
            max_length=self.max_question_length,
            truncation=True,
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

    @staticmethod
    def _preprocess_to_md(
        source: str | os.PathLike,
        tokenizer_fn: callable = None,
        cache_dir: str | os.PathLike | None = None,
    ) -> str | os.PathLike:
        """Preprocess the dataset to a markdown file.

        Args:
            source (str | os.PathLike): The source file or directory to preprocess.
        """
        source = Path(source)

        if source.is_dir():
            basename = get_index_basename()
            # filename = os.path.join(self.local, self.split, basename)  # pyright: ignore
            hashed_filename = source / basename
            if hashed_filename.exists():
                logger.info(f"Found existing index file {hashed_filename}")
                return source

        # No index.json file found, so we need to create it
        if cache_dir is None:
            cache_dir = GOLDENRETRIEVER_CACHE_DIR
        # check if cache dir exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        # get filename from the url
        hashed_filename = url_to_filename(str(source), None)
        # get cache path to put the file
        cache_path = cache_dir / hashed_filename

        # cache_path = str(cache_path)

        # the file is already here, return it
        if file_exists(cache_path):  # and not force_download:
            logger.info(
                f"{source} found in cache"  # , set `force_download=True` to force the download"
            )
            return str(cache_path)

        dataset = build_hf_dataset(
            dataset_name=str(source),
            data_subset=None,
            split="train",
            shuffle=False,
            is_local=True,
            # num_workers=args.num_workers,
        )
        num_workers = None
        if tokenizer_fn is not None:
            if num_workers is None:
                # Multiple workers is only supported on linux machines
                if "linux" or "macos" in platform.platform().lower():
                    num_workers = max(1, psutil.cpu_count())
                else:
                    num_workers = 0
            dataset = dataset.map(tokenizer_fn, desc="Tokenizing data")
        # dataloader = build_dataloader(
        #     dataset=dataset, batch_size=None, num_workers=None
        # )

        # def generate_samples(loader, tokenizer_fn=None):
        #     for batch in loader:
        #         if tokenizer_fn is not None:
        #             batch = tokenizer_fn(batch)
        #         yield batch
                # batch has key: list of values, we want to yield a list of dicts
                # keys = list(batch.keys())
                # current_bs = len(batch[keys[0]])
                # for idx in range(current_bs):
                #     yield {k: v[idx] if isinstance(v, list) else v for k, v in batch.items()}

        if tokenizer_fn is None:
            columns = {
                "id": "str",
                "question": "str",
                "positive_ctxs": "json",
                "negative_ctxs": "json",
                "hard_negative_ctxs": "json",
            }
        else:
            columns = {
                "id": "str",
                "question": "pkl",
                "passage": "pkl",
                "positive_pssgs": "pkl",
                "positives": "pkl",
                "negatives": "pkl",
                "hard_negatives": "pkl",
            }
        with MDSWriter(columns=columns, out=str(cache_path)) as out:
            for sample in tqdm(
                # generate_samples(dataloader, tokenizer_fn=tokenizer_fn),
                dataset,
                desc=f"Converting {source} to MDS",
            ):
                out.write(sample)

        return str(cache_path)


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
        positives_pssgs = split_fn(batch["positives_pssgs"])

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
                        positives_pssgs=positives_pssgs[i],
                        passages_ids=batch["passages_ids"],
                    )
                )
            )
        return batches

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

        passages_in_batch = {}
        for sample in batch:
            passages_in_batch.update(
                {tuple(passage["input_ids"]): passage for passage in sample["passage"]}
            )
            if "mined_passages" in sample:
                passages_in_batch.update(
                    {
                        tuple(passage["input_ids"]): passage
                        for passage in sample["mined_passages"]
                    }
                )

        batch = ModelInputs(
            dict(
                sample_idx=[s["id"] for s in batch],
                questions=[s["question"] for s in batch],
                passages=list(passages_in_batch.values()),
                passages_ids=set(passages_in_batch.keys()),
                # TODO: change the name of the two following keys
                positives_pssgs=[s["positive_pssgs"] for s in batch],
                positives=[s["positives"] for s in batch],
            )
        )

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
                "positives_pssgs": batch["positives_pssgs"],
                "passages_ids": batch["passages_ids"],
            }
        )
        return model_inputs
