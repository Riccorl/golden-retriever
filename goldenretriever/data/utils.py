import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import transformers as tr
from tqdm import tqdm

from goldenretriever.common.model_inputs import ModelInputs


class HardNegativesManager:
    def __init__(
        self,
        tokenizer: tr.PreTrainedTokenizer,
        data: Union[List[Dict], os.PathLike, Dict[int, List]] = None,
        max_length: int = 64,
        batch_size: int = 1000,
        lazy: bool = False,
    ) -> None:
        self._db: dict = None
        self.tokenizer = tokenizer

        if data is None:
            self._db = {}
        else:
            if isinstance(data, Dict):
                self._db = data
            elif isinstance(data, os.PathLike):
                with open(data) as f:
                    self._db = json.load(f)
            else:
                raise ValueError(
                    f"Data type {type(data)} not supported, only Dict and os.PathLike are supported."
                )
        # add the tokenizer to the class for future use
        self.tokenizer = tokenizer

        # invert the db to have a passage -> sample_idx mapping
        self._passage_db = defaultdict(set)
        for sample_idx, passages in self._db.items():
            for passage in passages:
                self._passage_db[passage].add(sample_idx)

        self._passage_hard_negatives = {}
        if not lazy:
            # create a dictionary of passage -> hard_negative mapping
            batch_size = min(batch_size, len(self._passage_db))
            unique_passages = list(self._passage_db.keys())
            for i in tqdm(
                range(0, len(unique_passages), batch_size),
                desc="Tokenizing Hard Negatives",
            ):
                batch = unique_passages[i : i + batch_size]
                tokenized_passages = self.tokenizer(
                    batch,
                    max_length=max_length,
                    truncation=True,
                )
                for i, passage in enumerate(batch):
                    self._passage_hard_negatives[passage] = {
                        k: tokenized_passages[k][i] for k in tokenized_passages.keys()
                    }

    def __len__(self) -> int:
        return len(self._db)

    def __getitem__(self, idx: int) -> Dict:
        return self._db[idx]

    def __iter__(self):
        for sample in self._db:
            yield sample

    def __contains__(self, idx: int) -> bool:
        return idx in self._db

    def get(self, idx: int) -> List[str]:
        """Get the hard negatives for a given sample index."""
        if idx not in self._db:
            raise ValueError(f"Sample index {idx} not in the database.")

        passages = self._db[idx]

        output = []
        for passage in passages:
            if passage not in self._passage_hard_negatives:
                self._passage_hard_negatives[passage] = self._tokenize(passage)
            output.append(self._passage_hard_negatives[passage])

        return output

    def _tokenize(self, passage: str) -> Dict:
        return self.tokenizer(passage, max_length=self.max_length, truncation=True)


class NegativeSampler:
    def __init__(
        self, num_elements: int, probabilities: Optional[Union[List, np.ndarray]] = None
    ):
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)

        if probabilities is None:
            # probabilities should sum to 1
            probabilities = np.random.random(num_elements)
            probabilities /= np.sum(probabilities)
        self.probabilities = probabilities

    def __call__(
        self,
        sample_size: int,
        num_samples: int = 1,
        probabilities: np.array = None,
        exclude: List[int] = None,
    ) -> np.array:
        """
        Fast sampling of `sample_size` elements from `num_elements` elements.
        The sampling is done by randomly shifting the probabilities and then
        finding the smallest of the negative numbers. This is much faster than
        sampling from a multinomial distribution.

        Args:
            sample_size (`int`):
                number of elements to sample
            num_samples (`int`, optional):
                number of samples to draw. Defaults to 1.
            probabilities (`np.array`, optional):
                probabilities of each element. Defaults to None.
            exclude (`List[int]`, optional):
                indices of elements to exclude. Defaults to None.

        Returns:
            `np.array`: array of sampled indices
        """
        if probabilities is None:
            probabilities = self.probabilities

        if exclude is not None:
            probabilities[exclude] = 0
            # re-normalize?
            # probabilities /= np.sum(probabilities)

        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        sampled_indices = np.argpartition(shifted_probabilities, sample_size, axis=1)[
            :, :sample_size
        ]
        return sampled_indices


def batch_generator(samples: Iterable[Any], batch_size: int) -> Iterable[Any]:
    """
    Generate batches from samples.

    Args:
        samples (`Iterable[Any]`): Iterable of samples.
        batch_size (`int`): Batch size.

    Returns:
        `Iterable[Any]`: Iterable of batches.
    """
    batch = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []

    # leftover batch
    if len(batch) > 0:
        yield batch


import math
from typing import TypeVar, Optional, Iterator

import torch
# from . import Sampler, Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, Sampler, SubsetRandomSampler


T_co = TypeVar("T_co", covariant=True)

class BatchNegatives(Sampler):
    pass
    # def __init__(self, dataset) #, batch_size, indices=None, shuffle=True):
        # self.batch_size = batch_size
        # self.shuffle = shuffle
        # # get the indicies and length
        # self.indices = [(i, src_len) for i, (src, src_len, trg, trg_len) in enumerate(dataset)]
        # # if indices are passed, then use only the ones passed (for ddp)
        # if indices is not None:
        #     self.indices = torch.tensor(self.indices)[indices].tolist()

    # def __iter__(self):
        # if self.shuffle:
        #     random.shuffle(self.indices)

        # pooled_indices = []
        # # create pool of indices with similar lengths
        # for i in range(0, len(self.indices), self.batch_size * 100):
        #     pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
        # self.pooled_indices = [x[0] for x in pooled_indices]

        # # yield indices for current batch
        # batches = [self.pooled_indices[i:i + self.batch_size] for i in
        #            range(0, len(self.pooled_indices), self.batch_size)]

        # if self.shuffle:
        #     random.shuffle(batches)
        # for batch in batches:
        #     yield batch


class GoldenDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        question_batch_size: int = 32,
        passage_batch_size: int = 400,
    ) -> None:
        super().__init__(self, dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.question_batch_size = question_batch_size
        self.passage_batch_size = passage_batch_size

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # return iter(self._create_batches(indices))

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def _create_batches(self, indices):
        batch = []
        batches = []
        passages_in_batch = {}
        for index in indices:
            sample = self.dataset[index]
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
                if len(batch) > self.question_batch_size:
                    for splited_batch in self.split_batch(
                        batch_dict, self.question_batch_size
                    ):
                        batches.append(splited_batch)
                else:
                    batches.append(batch_dict)

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
            # check for hard negatives and add with a probability of 0.1
            # if self.hn_manager is not None:
            #     if sample["id"] in self.hn_manager:
            #         passages_in_batch.update(
            #             {
            #                 tuple(passage["input_ids"]): passage
            #                 for passage in self.hn_manager.get(sample["id"])
            #             }
            #         )
            #     else:
            #         print(f"Sample {sample['id']} not in hn_manager")

        if len(batch) > 0:
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
            if len(batch) > self.question_batch_size:
                for splited_batch in self.split_batch(
                    batch_dict, self.question_batch_size
                ):
                    batches.append(splited_batch)
            else:
                batches.append(batch_dict)

        return batches

    @staticmethod
    def split_batch(
        batch: Union[Dict[str, Any], ModelInputs], microbatch_size: int
    ) -> List[ModelInputs]:
        """
        Split a batch into multiple batches of size `question_batch_size` while keeping
        the same number of passages.
        """

        def split_fn(x):
            return [
                x[i : i + microbatch_size] for i in range(0, len(x), microbatch_size)
            ]

        # split the sample_idx
        sample_idx = split_fn(batch["sample_idx"])
        # split the questions
        questions = split_fn(batch["questions"])
        # split the positives
        positives = split_fn(batch["positives"])
        # split the positives_pssgs
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
                        positives_pssgs=positives_pssgs[i],
                    )
                )
            )
        return batches

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
