import json
import os
from functools import partial
from typing import Any, Dict, List, Union, Optional

import numpy as np
import psutil
import torch
import transformers as tr

from data.labels import Labels
from data.sampler import NegativeSampler
from utils.logging import get_console_logger

logger = get_console_logger()


class DPRMixin:
    @property
    def contexts(self):
        return list(self.context_manager.get_labels().keys())

    def shuffle_data(self, seed: int = 42):
        self.data = self.data.shuffle(seed=seed)

    @staticmethod
    def _process_dataset_sample(
        sample: Dict,
        tokenizer: tr.PreTrainedTokenizer,
        max_positives: int,
        max_negatives: int,
        max_hard_negatives: int,
        max_contexts: int = -1,
        max_question_length: int = 256,
        max_context_length: int = 128,
    ):
        positive_ctxs = [p["text"].strip() for p in sample["positive_ctxs"]]
        if max_positives != -1:
            positive_ctxs = positive_ctxs[:max_positives]
        negative_ctxs = [n["text"].strip() for n in sample["negative_ctxs"]]
        if max_negatives != -1:
            negative_ctxs = negative_ctxs[:max_negatives]
        hard_negative_ctxs = [h["text"].strip() for h in sample["hard_negative_ctxs"]]
        if max_hard_negatives != -1:
            hard_negative_ctxs = hard_negative_ctxs[:max_hard_negatives]

        question = tokenizer(
            sample["question"], max_length=max_question_length, truncation=True
        )
        positive_ctxs = [
            tokenizer(p, max_length=max_context_length, truncation=True)
            for p in positive_ctxs
        ]
        negative_ctxs = [
            tokenizer(n, max_length=max_context_length, truncation=True)
            for n in negative_ctxs
        ]
        hard_negative_ctxs = [
            tokenizer(h, max_length=max_context_length, truncation=True)
            for h in hard_negative_ctxs
        ]

        context = positive_ctxs + negative_ctxs + hard_negative_ctxs
        if max_contexts != -1:
            context = context[:max_contexts]
        output = {
            "question": question,
            "context": context,
            "positives": set([p["text"].strip() for p in sample["positive_ctxs"]]),
            "positive_ctxs": positive_ctxs,
            "negative_ctxs": negative_ctxs,
            "hard_negative_ctxs": hard_negative_ctxs,
            "positive_index_end": len(positive_ctxs),
        }
        return output

    def _sample_dataset_negatives(
        self,
        data,
        tokenizer: tr.PreTrainedTokenizer,
        context_manager: Labels,
        sample_by_frequency: bool = True,
        max_negatives_to_sample: int = 64,
        *args,
        **kwargs,
    ) -> Any:
        if sample_by_frequency:
            logger.log("Computing contexts frequencies")
            context_frequencies = self.compute_contexts_frequency(data, context_manager)
            # get only those contexts that have frequency > 0
            context_idx_above_zero = [
                idx for idx, freq in enumerate(context_frequencies) if freq > 0
            ]
            # build a reduced context_manager for sampling negatives
            sampled_context_manager = Labels()
            sampled_context_manager.add_labels(
                [
                    context_manager.get_label_from_index(idx)
                    for idx in context_idx_above_zero
                ]
            )
            context_frequencies = self.compute_contexts_frequency(
                data, sampled_context_manager
            )
        else:
            context_frequencies = None
            sampled_context_manager = context_manager
        sample_space_size = sampled_context_manager.get_label_size()
        logger.log(f"Sampling negative contexts from {sample_space_size} samples")
        # update the samples with the sampled negatives
        data = data.map(
            partial(
                self._sample_negatives,
                tokenizer=tokenizer,
                sample_space_size=sample_space_size,
                context_frequencies=context_frequencies,
                context_manager=sampled_context_manager,
                max_negatives_to_sample=max_negatives_to_sample,
            ),
            keep_in_memory=True,
            num_proc=psutil.cpu_count(),
        )
        return data

    @staticmethod
    def _sample_negatives(
        sample: Dict[str, Any],
        tokenizer: tr.PreTrainedTokenizer,
        context_manager: Labels,
        sample_space_size: int,
        context_frequencies: np.array,
        max_negatives_to_sample: int = 0,
        max_context_length: int = 128,
    ):
        """
        Sample negatives and add them to the sample.
        """

        # TODO: make faster sampling
        negative_sampler = NegativeSampler(sample_space_size, context_frequencies)

        positives_contexts_ids = sample["positive_ctxs"]
        negative_contexts_ids = sample["negative_ctxs"]
        hard_negative_contexts_ids = sample["hard_negative_ctxs"]

        positives = sample["positives"]
        positive_indices = [context_manager.get_index_from_label(p) for p in positives]

        actual_number_of_contexts = (
            len(positives_contexts_ids)
            + len(negative_contexts_ids)
            + len(hard_negative_contexts_ids)
        )

        sampled_negative_contexts = []
        if max_negatives_to_sample > 0:
            if actual_number_of_contexts < max_negatives_to_sample:
                remaining_contexts = max_negatives_to_sample - actual_number_of_contexts
                sampled = negative_sampler(remaining_contexts, exclude=positive_indices)
                sampled_negative_contexts += [
                    context_manager.get_label_from_index(s) for s in sampled[0]
                ]

        sampled_negative_ids = [
            tokenizer(n, max_length=max_context_length, truncation=True)
            for n in sampled_negative_contexts
        ]
        negative_contexts_ids += sampled_negative_ids
        context = (
            positives_contexts_ids + negative_contexts_ids + hard_negative_contexts_ids
        )
        sample["context"] = context
        return sample

    @staticmethod
    def compute_contexts_frequency(data, context_manager) -> np.array:
        """
        Compute the frequency of each context in the dataset.
        """
        if context_manager is None:
            raise ValueError(
                "Context manager is required to compute the frequency of contexts"
            )
        frequency = np.zeros(context_manager.get_label_size())
        for sample in data:
            for context in sample["positives"]:
                contex_index = context_manager.get_index_from_label(context)
                frequency[contex_index] += 1
        # normalize the frequency
        frequency = frequency / frequency.sum()
        return frequency

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

    def add_fields_to_samples(
        self,
        updates: Dict[int, Dict[str, Any]],
    ):
        """
        Update the data with the updates.

        Args:
            updates (:obj:`Dict[int, Dict[str, Any]]`):
                Dictionary of updates to apply to the data. The key is the index of the sample to update.
        """

        def update_fn(sample):
            sample.update(updates[sample["sample_idx"]])
            return sample

        self.data = self.data.map(update_fn)
