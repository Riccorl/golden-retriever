import concurrent.futures
import json
import os
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import psutil
import torch
import transformers as tr
from datasets import Dataset, IterableDataset, load_dataset
from numpy.random import choice
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from goldenretriever.common.log import get_console_logger, get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.dpr.hard_negatives_manager import HardNegativeManager
from goldenretriever.data.labels import Labels


class NegativesStrategy:
    def __init__(
        self, pad_token_id: int, pad_attention_id: int = 0, pad_token_type_id: int = 0
    ) -> None:
        # define padding operations for each input
        self.padding_ops = {
            "input_ids": partial(
                self.pad_sequence,
                value=pad_token_id,
            ),
            "attention_mask": partial(self.pad_sequence, value=pad_attention_id),
            "token_type_ids": partial(
                self.pad_sequence,
                value=pad_token_type_id,
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

    def batch_fn(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def collate_fn(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class InBatchNegativesStrategy(NegativesStrategy):
    def batch_fn(
        self,
        batched_samples,
        hard_negatives_manager: HardNegativeManager,
        context_batch_size,
        question_batch_size,
    ) -> Dict[str, List[Dict[str, Any]]]:
        batch = []
        contexts_in_batch = set()
        output_batches = {"batches": []}
        batched_samples_list = []
        for i in range(len(batched_samples["question"])):
            batched_samples_list.append(
                {k: batched_samples[k][i] for k in batched_samples.keys()}
            )
        for sample in batched_samples_list:
            if len(contexts_in_batch) >= context_batch_size:
                output_batches["batches"].append(batch)

                # reset batch
                batch = []
                contexts_in_batch = set()

            batch.append(sample)
            for context_type in {
                "positive_ctxs",
                "negative_ctxs",
                "sampled_negative_ctxs",
                "hard_negative_ctxs",
                "retrieved_hard_negatives",
            }:
                # yes it's a bit ugly but it works :)
                # count the number of contexts in the batch and stop if we reach the limit
                # we use a set to avoid counting the same context twice
                # we use a tuple because set doesn't support lists
                # we use input_ids as discriminator
                contexts_in_batch |= set(
                    tuple(s["input_ids"])
                    for sample in batch
                    if context_type in sample
                    for s in sample[context_type]
                )
                # also add the contexts from the hard negatives dict
                if hard_negatives_manager is not None:
                    contexts_in_batch |= set(
                        tuple(s["input_ids"])
                        for sample in batch
                        if sample["sample_idx"] in hard_negatives_manager
                        for s in hard_negatives_manager.get(sample["sample_idx"])
                    )
        if len(batch) > 0:
            output_batches["batches"].append(batch)

        return output_batches

    def collate_fn(
        self,
        batch,
        hard_negatives_manager: Optional[HardNegativeManager] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        batch = batch["batches"]

        # get data from batch
        questions = [sample["question"] for sample in batch]
        positives = [sample["positives"] for sample in batch]
        sample_idxs = [sample["sample_idx"] for sample in batch]

        # this is needed to get the correct labels for each question
        positives_ctxs = [sample["positive_ctxs"] for sample in batch]
        negatives_ctxs = [sample["negative_ctxs"] for sample in batch]
        if "sampled_negative_ctxs" in batch[0]:
            negatives_ctxs += [sample["sampled_negative_ctxs"] for sample in batch]

        hard_negatives_ctxs = [sample["hard_negative_ctxs"] for sample in batch]

        if hard_negatives_manager is not None:
            for sample_idx in sample_idxs:
                hard_negatives_ctxs.append(hard_negatives_manager.get(sample_idx))

        # convert the questions to a batch
        questions = self.convert_to_batch(questions)

        # now we need to make the batch of contexts
        # it can happen that there are duplicate contexts from different questions
        # so we need to remove them
        flat_positives = [p for ps in positives_ctxs for p in ps]
        flat_negatives = [n for ns in negatives_ctxs for n in ns]
        flat_hard_negatives = [hn for hns in hard_negatives_ctxs for hn in hns]

        # remove duplicates based on input_ids (input_ids is a list of int)
        flat_positives = list(
            {tuple(p["input_ids"]): p for p in flat_positives}.values()
        )
        flat_negatives = list(
            {tuple(n["input_ids"]): n for n in flat_negatives}.values()
        )
        flat_hard_negatives = list(
            {tuple(hn["input_ids"]): hn for hn in flat_hard_negatives}.values()
        )
        unique_contexts = flat_positives + flat_negatives + flat_hard_negatives
        contexts = self.convert_to_batch(unique_contexts)
        # build an index to map the position of the context in the batch
        context_index = {
            tuple(c["input_ids"]): i for i, c in enumerate(unique_contexts)
        }

        # now we can create the labels
        labels = torch.zeros(
            questions["input_ids"].shape[0], contexts["input_ids"].shape[0]
        )
        # iterate over the questions and set the labels to 1 if the context is positive
        for sample_idx in range(len(questions["input_ids"])):
            for ctx in positives_ctxs[sample_idx]:
                # get the index of the positive context
                index = context_index[tuple(ctx["input_ids"])]
                # set the label to 1
                labels[sample_idx, index] = 1

        model_inputs = {
            "questions": questions,
            "contexts": contexts,
            "labels": labels,
            "positives": positives,
            "sample_idx": [sample["sample_idx"] for sample in batch],
        }
        return model_inputs


class HardNegativeManager:
    def __init__(
        self,
        tokenizer: tr.PreTrainedTokenizer,
        data: Union[List[Dict], os.PathLike, Dict[int, List]] = None,
        max_length: int = 64,
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

        # invert the db to have a context -> sample_idx mapping
        self._context_db = defaultdict(set)
        for sample_idx, contexts in self._db.items():
            for context in contexts:
                self._context_db[context].add(sample_idx)

        self._context_hard_negatives = {}
        if not lazy:
            # create a dictionary of context -> hard_negative mapping
            tokenized_contexts = self.tokenizer(
                list(self._context_db.keys()), max_length=max_length, truncation=True
            )
            for i, context in enumerate(self._context_db):
                self._context_hard_negatives[context] = {
                    k: tokenized_contexts[k][i] for k in tokenized_contexts.keys()
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

        contexts = self._db[idx]

        output = []
        for context in contexts:
            if context not in self._context_hard_negatives:
                self._context_hard_negatives[context] = self._tokenize(context)
            output.append(self._context_hard_negatives[context])

        return output

    def _tokenize(self, context: str) -> Dict:
        return self.tokenizer(context, max_length=self.max_length, truncation=True)
