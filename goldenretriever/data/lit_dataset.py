# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import hashlib
import os
from logging import Logger
from time import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from litdata import StreamingDataset
import numpy as np
import torch
from torch.utils.data import IterableDataset

from litdata.constants import (
    _DEFAULT_CACHE_DIR,
    _INDEX_FILENAME,
)
from litdata.streaming import Cache
from litdata.streaming.item_loader import BaseItemLoader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer
from litdata.streaming.shuffle import FullShuffle, NoShuffle, Shuffle
from litdata.utilities.env import _DistributedEnv, _is_in_dataloader_worker, _WorkerEnv
from litdata.streaming.dataset import (
    _associate_chunks_to_workers,
    _replay_chunks_sampling,
    _replay_sampling,
    _should_replace_path,
    _try_create_cache_dir,
)

from litdata.constants import (
    _TORCH_DTYPES_MAPPING,
    _TORCH_GREATER_EQUAL_2_1_0,
)

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import PyTree, tree_unflatten

from transformers import PreTrainedTokenizerBase
from time import sleep

logger = Logger(__name__)


class GoldenLoader(BaseItemLoader):
    """The Pytree Loader is the default loader of the Cache object."""

    def __init__(
        self,
        question_tokenizer,
        passage_tokenizer=None,
        max_positives: int = -1,
        max_negatives: int = -1,
        max_hard_negatives: int = -1,
        max_passages: int = -1,
        max_question_length: int = 40,
        max_passage_length: int = 40,
        metadata_fields: Optional[Sequence[str]] = None,
        metadata_separator: str = "\t",
    ) -> None:
        self.question_tokenizer = question_tokenizer
        self.passage_tokenizer = question_tokenizer
        self._chunk_filepaths: Dict[str, bool] = {}

        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_passages = max_passages
        self.max_question_length = max_question_length
        self.max_passage_length = max_passage_length
        self.metadata_fields = metadata_fields
        self.metadata_separator = metadata_separator

    def generate_intervals(self) -> List[Tuple[int, int]]:
        intervals = []
        begin = 0
        end = 0
        for chunk in self._chunks:
            end += chunk["chunk_size"]
            intervals.append((begin, end))
            begin += chunk["chunk_size"]
        return intervals

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        pass

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        chunk_bytes: int,
    ) -> bytes:
        offset = (1 + (index - begin) if index >= begin else index + 1) * 4

        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(
            chunk_filepath
        ):
            del self._chunk_filepaths[chunk_filepath]

        if chunk_filepath not in self._chunk_filepaths:
            exists = (
                os.path.exists(chunk_filepath)
                and os.stat(chunk_filepath).st_size >= chunk_bytes
            )

            while not exists:
                sleep(0.1)
                exists = (
                    os.path.exists(chunk_filepath)
                    and os.stat(chunk_filepath).st_size >= chunk_bytes
                )

            self._chunk_filepaths[chunk_filepath] = True

        with open(chunk_filepath, "rb", 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
            fp.seek(begin)
            data = fp.read(end - begin)
        
        deserialized = self.deserialize(data)
        if isinstance(deserialized["question"], str):
            return self._tokenize(deserialized)
        else:
            return deserialized
        # return deserialized

    @functools.lru_cache(maxsize=128)
    def _data_format_to_key(self, data_format: str) -> str:
        if ":" in data_format:
            serialier, serializer_sub_type = data_format.split(":")
            if serializer_sub_type in self._serializers:
                return serializer_sub_type
            return serialier
        return data_format

    def deserialize(self, raw_item_data: bytes) -> "PyTree":
        """Deserialize the raw bytes into their python equivalent."""
        idx = len(self._config["data_format"]) * 4
        sizes = np.frombuffer(raw_item_data[:idx], np.uint32)
        data = []
        for size, data_format in zip(sizes, self._config["data_format"]):
            serializer = self._serializers[self._data_format_to_key(data_format)]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config["data_spec"])

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if os.path.exists(chunk_filepath):
            os.remove(chunk_filepath)

    def _get_passages(self, passages: List[Dict[str, str]]) -> List[str]:
        formatted_passages = []
        for passage in passages:
            formatted_passage = passage["text"]
            if self.metadata_fields is not None:
                metadata = self.metadata_separator.join(
                    [passage.get(field, "") for field in self.metadata_fields]
                )
                formatted_passage = (
                    f"{formatted_passage}{self.metadata_separator}{metadata}"
                )
            formatted_passages.append(formatted_passage)

        # remove duplicates
        formatted_passages = list(set(formatted_passages))
        return formatted_passages

    # How to tokenize a text sample to a token sample
    def _tokenize(self, sample) -> Dict[str, List[int]]:
        # remove duplicates and limit the number of passages
        # positives = list(set([p["text"] for p in sample["positive_ctxs"]]))
        positives = self._get_passages(sample["positive_ctxs"])
        if self.max_positives != -1:
            positives = positives[: self.max_positives]

        # negatives = list(set([n["text"] for n in sample["negative_ctxs"]]))
        negatives = self._get_passages(sample["negative_ctxs"])
        if self.max_negatives != -1:
            negatives = negatives[: self.max_negatives]

        # hard_negatives = list(set([h["text"] for h in sample["hard_negative_ctxs"]]))
        hard_negatives = self._get_passages(sample["hard_negative_ctxs"])
        if self.max_hard_negatives != -1:
            hard_negatives = hard_negatives[: self.max_hard_negatives]

        text_pair = sample.get("doc_topic", None)
        question = self.question_tokenizer(
            sample["question"],
            text_pair=text_pair,
            max_length=self.max_question_length,
            truncation=True,
        )

        passage = positives + negatives + hard_negatives
        if self.max_passages != -1:
            passage = passage[: self.max_passages]

        passage = self.passage_tokenizer(
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

    # def __init__(self, tokenizer: PreTrainedTokenizerBase):
    #     """The Tokens Loader is an optimizer item loader for NLP.

    #     Arguments:
    #         block_size: The context length to use during training.

    #     """

    #     super().__init__()
    #     # self._block_size = block_size
    #     self._tokenizer = tokenizer
    #     self._passage_batch_size = 8
    #     self._mmaps: Dict[int, np.memmap] = {}
    #     self._buffers: Dict[int, bytes] = {}
    #     self._dtype: Optional[torch.dtype] = None
    #     self._chunk_filepaths: Dict[str, bool] = {}

    # def state_dict(self) -> Dict:
    #     return {
    #         "tokenizer": self._tokenizer.name_or_path,
    #     }

    # def setup(self, config: Dict, chunks: List, serializers: Dict[str, Serializer]) -> None:
    #     super().setup(config, chunks, serializers)
    #     self._dtype = torch.float32 #_TORCH_DTYPES_MAPPING[int(config["data_format"][0].split(":")[1])]
    #     if all(chunk["dim"] is None for chunk in self._chunks):
    #         raise ValueError("The provided chunks isn't properly setup.")

    # def generate_intervals(self) -> List[Tuple[int, int]]:
    #     intervals = []
    #     # begin = 0
    #     # end = 0
    #     # for chunk in self._chunks:
    #     #     dim = chunk["dim"]
    #     #     num_blocks = dim // self._block_size
    #     #     end += num_blocks
    #     #     intervals.append((begin, end))
    #     #     begin += num_blocks
    #     return intervals

    # def _load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
    #     if chunk_index in self._mmaps:
    #         return
    #     chunk = self._chunks[chunk_index]

    #     # Skip the header
    #     # The number of items + the number of offsets (number of items in the chunk + 1)
    #     # multiplied by the header encoding dtype (np.uint32)
    #     offset = (1 + chunk["chunk_size"] + 1) * 4
    #     mmap = np.memmap(chunk_filepath, mode="r", order="C", offset=offset)
    #     self._mmaps[chunk_index] = mmap
    #     self._buffers[chunk_index] = memoryview(mmap)  # type: ignore

    # def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
    #     # This is called within the prepare chunks thread, so we overlap data loading with data reading.
    #     if chunk_filepath not in self._chunk_filepaths:
    #         self._chunk_filepaths[chunk_filepath] = True

    #     if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0:
    #         self._load_chunk(chunk_index, chunk_filepath)

    # def load_item_from_chunk(
    #     self, index: int, chunk_index: int, chunk_filepath: str, begin: int, chunk_bytes: int
    # ) -> torch.Tensor:
    #     if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
    #         del self._chunk_filepaths[chunk_filepath]

    #     if chunk_filepath not in self._chunk_filepaths:
    #         exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0

    #         while not exists:
    #             sleep(0.1)
    #             exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0

    #         self._chunk_filepaths[chunk_filepath] = True

    #     self._load_chunk(chunk_index, chunk_filepath)
    #     assert self._dtype

    #     buffer: bytes = self._buffers[chunk_index]
    #     offset = self._dtype.itemsize * (index - begin) * self._block_size
    #     return torch.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)

    # def delete(self, chunk_index: int, chunk_filepath: str) -> None:
    #     if os.path.exists(chunk_filepath):
    #         if chunk_index in self._buffers:
    #             del self._buffers[chunk_index]
    #         if chunk_index in self._mmaps:
    #             del self._mmaps[chunk_index]
    #         os.remove(chunk_filepath)


class GoldenStreamingDataset(StreamingDataset):
    """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class."""

    def __init__(
        self,
        name: str,
        question_tokenizer: PreTrainedTokenizerBase,
        input_dir: Union[str, "Dir"],
        passage_tokenizer: PreTrainedTokenizerBase | None = None,
        item_loader: Optional[BaseItemLoader] = None,
        shuffle: bool = False,
        drop_last: Optional[bool] = None,
        seed: int = 42,
        serializers: Optional[Dict[str, Serializer]] = None,
        max_cache_size: Union[int, str] = "100GB",
        # golden retriever specific
        preprocess: bool = False,
        max_positives: int = -1,
        max_negatives: int = -1,
        max_hard_negatives: int = -1,
        max_passages: int = -1,
        max_question_length: int = 40,
        max_passage_length: int = 40,
        metadata_fields: Optional[Sequence[str]] = None,
        metadata_separator: str = "\t",
    ) -> None:
        """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class.

        Arguments:
            input_dir: Path to the folder where the input data is stored.
            item_loader: The logic to load an item from a chunk.
            shuffle: Whether to shuffle the data.
            drop_last: If `True`, drops the last items to ensure that
                all processes/workers return the same amount of data.
                The argument `drop_last` is set to `True` in a distributed setting
                and `False` otherwise.
            seed: Random seed for shuffling.
            serializers: The serializers used to serialize and deserialize the chunks.
            max_cache_size: The maximum cache size used by the StreamingDataset.

        """

        # we initialize subclass specific attributes first
        # because we need to use them in case of preprocessing
        self.name = name
        self.question_tokenizer = question_tokenizer
        self.passage_tokenizer = passage_tokenizer or self.question_tokenizer
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_passages = max_passages
        self.max_question_length = max_question_length
        self.max_passage_length = max_passage_length
        self.metadata_fields = metadata_fields
        self.metadata_separator = metadata_separator

        if item_loader is None:
            item_loader = GoldenLoader(
                question_tokenizer=self.question_tokenizer, passage_tokenizer=self.passage_tokenizer,
            )

        super().__init__(
            input_dir=input_dir,
            item_loader=item_loader,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            serializers=serializers,
            max_cache_size=max_cache_size,
        )

    def set_shuffle(self, shuffle: bool) -> None:
        self.shuffle = shuffle

    def set_epoch(self, current_epoch: int) -> None:
        """Set the current epoch to the dataset on epoch starts.

        When using the StreamingDataLoader, this is done automatically

        """
        # If the state dict has been reloaded, don't override the current epoch
        # The StreamingDataloader would clean this out
        if self._state_dict is None:
            self.current_epoch = current_epoch

    def _create_cache(self, worker_env: _WorkerEnv) -> Cache:
        if _should_replace_path(self.input_dir.path):
            cache_path = _try_create_cache_dir(
                input_dir=(
                    self.input_dir.path if self.input_dir.path else self.input_dir.url
                )
            )
            if cache_path is not None:
                self.input_dir.path = cache_path

        cache = Cache(
            input_dir=self.input_dir,
            item_loader=self.item_loader,
            chunk_bytes=1,
            serializers=self.serializers,
            max_cache_size=self.max_cache_size,
        )
        cache._reader._try_load_config()

        if not cache.filled:
            raise ValueError(
                f"The provided dataset `{self.input_dir}` doesn't contain any {_INDEX_FILENAME} file."
                " HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
            )

        return cache

    def _create_shuffler(self, cache: Cache) -> Shuffle:
        seed = self.seed
        drop_last = self.drop_last
        if self._state_dict is not None:
            state: Dict[str, Any] = self._state_dict
            seed = state["seed"]
            drop_last = state["drop_last"]
        return (
            FullShuffle(cache, seed, drop_last)
            if self.shuffle
            else NoShuffle(cache, seed, drop_last)
        )

    def __len__(self) -> int:
        if self.shuffler is None:
            cache = self._create_cache(worker_env=_WorkerEnv.detect())
            self.shuffler = self._create_shuffler(cache)
        return self.shuffler.get_len(self.distributed_env, self.current_epoch)

    def __iter__(self) -> "StreamingDataset":
        # When the StreamingDataset is used within map or optimize, let's refetch the distributed env.
        if os.getenv("DATA_OPTIMIZER_GLOBAL_RANK"):
            self.distributed_env = _DistributedEnv.detect()

        self.worker_env = _WorkerEnv.detect()
        self.cache = self._create_cache(worker_env=self.worker_env)
        self.shuffler = self._create_shuffler(self.cache)

        # Handle restart
        if self._state_dict:
            self._validate_state_dict()
            state: Dict[str, Any] = self._state_dict
            self.current_epoch = state["current_epoch"]

        chunks_per_replica, intervals_per_replica = (
            self.shuffler.get_chunks_and_intervals_per_ranks(
                self.distributed_env, self.current_epoch
            )
        )
        chunks_replica = chunks_per_replica[
            self.distributed_env.global_rank % self.distributed_env.world_size
        ]
        intervals_replica = intervals_per_replica[
            self.distributed_env.global_rank % self.distributed_env.world_size
        ]

        # Handle restart
        if self._state_dict:
            self._resume(chunks_replica, intervals_replica)
        else:
            chunks_per_replica, intervals_per_replica = (
                self.shuffler.get_chunks_and_intervals_per_ranks(
                    self.distributed_env, self.current_epoch
                )
            )
            chunks_replica = chunks_per_replica[
                self.distributed_env.global_rank % self.distributed_env.world_size
            ]
            intervals_replica = intervals_per_replica[
                self.distributed_env.global_rank % self.distributed_env.world_size
            ]

            self.worker_chunks = []
            self.worker_intervals = []

            for i, (chunk_index, chunk_interval) in enumerate(
                zip(chunks_replica, intervals_replica)
            ):
                if i % self.worker_env.world_size != self.worker_env.rank:
                    continue
                self.worker_chunks.append(chunk_index)
                self.worker_intervals.append(chunk_interval)

            self.num_chunks = len(self.worker_chunks)

            self.current_indexes = []
            self.chunk_index = 0
            self.global_index = 0
            self.index = 0

        self.has_triggered_download = False
        self.last_time = time()

        return self

    def _get_passages(self, passages: List[Dict[str, str]]) -> List[str]:
        formatted_passages = []
        for passage in passages:
            formatted_passage = passage["text"]
            if self.metadata_fields is not None:
                metadata = self.metadata_separator.join(
                    [passage.get(field, "") for field in self.metadata_fields]
                )
                formatted_passage = (
                    f"{formatted_passage}{self.metadata_separator}{metadata}"
                )
            formatted_passages.append(formatted_passage)

        # remove duplicates
        formatted_passages = list(set(formatted_passages))
        return formatted_passages

    # How to tokenize a text sample to a token sample
    def _tokenize(self, sample) -> Dict[str, List[int]]:
        # remove duplicates and limit the number of passages
        # positives = list(set([p["text"] for p in sample["positive_ctxs"]]))
        positives = self._get_passages(sample["positive_ctxs"])
        if self.max_positives != -1:
            positives = positives[: self.max_positives]

        # negatives = list(set([n["text"] for n in sample["negative_ctxs"]]))
        negatives = self._get_passages(sample["negative_ctxs"])
        if self.max_negatives != -1:
            negatives = negatives[: self.max_negatives]

        # hard_negatives = list(set([h["text"] for h in sample["hard_negative_ctxs"]]))
        hard_negatives = self._get_passages(sample["hard_negative_ctxs"])
        if self.max_hard_negatives != -1:
            hard_negatives = hard_negatives[: self.max_hard_negatives]

        text_pair = sample.get("doc_topic", None)
        question = self.question_tokenizer(
            sample["question"],
            text_pair=text_pair,
            max_length=self.max_question_length,
            truncation=True,
        )

        passage = positives + negatives + hard_negatives
        if self.max_passages != -1:
            passage = passage[: self.max_passages]

        passage = self.passage_tokenizer(
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

    def _resume(self, chunks_replica: List[int], intervals_replica: List[Any]) -> None:
        assert self._state_dict
        assert self.worker_env
        assert self.shuffler

        state: Dict[str, Any] = self._state_dict

        num_workers = state["num_workers"]
        batch_size = state["batch_size"]

        # TODO: Implement elastic sampling where the number of workers, ranks can change.
        num_samples_yielded = self._state_dict["num_samples_yielded"]

        # replay sampling from each worker / chunks using the batch size
        workers_chunks, workers_intervals = _associate_chunks_to_workers(
            num_workers, self.worker_env, chunks_replica, intervals_replica
        )
        indexes = _replay_sampling(num_samples_yielded, batch_size, num_workers)
        chunks_index, indexes = _replay_chunks_sampling(workers_intervals, indexes)

        # select the chunks and intervals associated to this worker
        worker_rank = self.worker_env.rank
        self.num_chunks = len(workers_intervals[worker_rank])
        self.chunk_index = chunks_index[worker_rank]
        self.worker_chunks = workers_chunks[worker_rank]
        self.worker_intervals = workers_intervals[worker_rank]

        # replay the indexes for the current chunks
        interval = self.worker_intervals[self.chunk_index]
        current_indexes = np.arange(interval[0], interval[1])

        # re-shuffle the indexes
        current_indexes = self.shuffler(
            current_indexes, self.num_chunks, self.current_epoch, self.chunk_index
        )

        # skip any indexes already consumed
        current_indexes = current_indexes[indexes[worker_rank] :]
        self.current_indexes = current_indexes

        self.global_index = num_samples_yielded

        # bump the chunk_index
        self.chunk_index += 1

    def __getitem__(self, index: Union[ChunkedIndex, int]) -> Any:
        if self.cache is None:
            self.worker_env = _WorkerEnv.detect()
            self.cache = self._create_cache(worker_env=self.worker_env)
            self.shuffler = self._create_shuffler(self.cache)
        if isinstance(index, int):
            index = ChunkedIndex(index, self.cache._get_chunk_index_from_index(index))
        return self.cache[index]

    def __next__(self) -> Any:
        # Prevent to create more batch on a given process
        if self.global_index >= len(self):
            self.current_epoch += 1
            raise StopIteration

        # Lazily re-populate the interval to reduce memory usage.
        if len(self.current_indexes) == 0:
            if self.chunk_index == self.num_chunks:
                self.current_epoch += 1
                raise StopIteration

            # reset index
            self.index = 0

            interval = self.worker_intervals[self.chunk_index]
            current_indexes = np.arange(interval[0], interval[1])

            assert self.shuffler is not None
            assert self.num_chunks is not None
            self.current_indexes = self.shuffler(
                current_indexes, self.num_chunks, self.current_epoch, self.chunk_index
            )

            self.chunk_index += 1

        # Get the first index
        index = self.current_indexes.pop(0)

        # Call the `__getitem__` method.
        data = self.__getitem__(
            ChunkedIndex(
                index=index,
                chunk_index=self.worker_chunks[self.chunk_index - 1],
                # We provide the chunks indexes only one the first
                chunk_indexes=(
                    None if self.has_triggered_download else self.worker_chunks
                ),
                is_last_index=(self.chunk_index - 1) == len(self.worker_intervals)
                and len(self.current_indexes) == 1,
            )
        )

        self.has_triggered_download = True
        self.global_index += 1
        self.index += 1

        return data

    def state_dict(
        self, num_samples_yielded: int, num_workers: int, batch_size: int
    ) -> Dict[str, Any]:
        if _is_in_dataloader_worker():
            raise RuntimeError(
                "The method `state_dict` should only be called in the main process."
            )

        if self._state_dict is not None:
            self._state_dict["num_samples_yielded"] = num_samples_yielded
            return self._state_dict

        return {
            "num_samples_yielded": num_samples_yielded,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "current_epoch": self.current_epoch,
            "input_dir_path": self.input_dir.path,
            "input_dir_url": self.input_dir.url,
            "item_loader": self.item_loader.state_dict() if self.item_loader else None,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "world_size": self.distributed_env.world_size,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict:
            # the state is restored within the workers
            self._state_dict = state_dict

    def _validate_state_dict(self) -> None:
        assert self._state_dict
        assert self.worker_env
        assert self.cache

        state: Dict[str, Any] = self._state_dict

        if state["shuffle"] != self.shuffle:
            raise ValueError(
                "The provided `shuffle` state doesn't match the current one. "
                f"Found `{self.shuffle}` instead of `{state['shuffle']}`."
            )

        if state["num_workers"] != self.worker_env.world_size:
            raise ValueError(
                "The provided `num_workers` state doesn't match the current one. "
                f"Found `{self.worker_env.world_size}` instead of `{state['num_workers']}`."
            )

        # Note: We need to check whether the path has been resolved to its associated cache.
        # In this case, validate the cache folder is the same.
        if _should_replace_path(state["input_dir_path"]):
            cache_path = _try_create_cache_dir(
                input_dir=(
                    state["input_dir_path"]
                    if state["input_dir_path"]
                    else state["input_dir_url"]
                )
            )
            if cache_path != self.input_dir.path:
                raise ValueError(
                    "The provided `input_dir` path state doesn't match the current one. "
                    f"Found `{self.input_dir.path}` instead of `{cache_path}`."
                )
        elif state["input_dir_path"] != self.input_dir.path:
            raise ValueError(
                "The provided `input_dir` path state doesn't match the current one. "
                f"Found `{self.input_dir.path}` instead of `{state['input_dir_path']}`."
            )

        if state["input_dir_url"] != self.input_dir.url:
            raise ValueError(
                "The provided `input_dir` URL state doesn't match the current one. "
                f"Found `{self.input_dir.url}` instead of `{state['input_dir_url']}`."
            )

        if state["seed"] != self.seed:
            raise ValueError(
                "The provided `seed` state doesn't match the current one. "
                f"Found `{self.seed}` instead of `{state['seed']}`."
            )

        if self.item_loader and state["item_loader"] != self.item_loader.state_dict():
            raise ValueError(
                "The provided `item_loader` state doesn't match the current one. "
                f"Found `{self.item_loader.state_dict()}` instead of `{state['item_loader']}`."
            )

        if state["drop_last"] != self.drop_last:
            raise ValueError(
                "The provided `drop_last` state doesn't match the current one. "
                f"Found `{self.drop_last}` instead of `{state['drop_last']}`."
            )
