import json
import os
import time
from functools import partial
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union, Optional, Callable

import numpy as np
import psutil
import torch
import transformers as tr
from datasets import load_dataset
from numpy.random import choice
from tqdm import tqdm

from golden_retriever.common.log import get_console_logger
from golden_retriever.common.model_inputs import ModelInputs
from golden_retriever.data.datasets import GenerativeDataset, BaseDataset
from golden_retriever.data.labels import Labels

logger = get_console_logger()


class DPRMixin:
    @staticmethod
    def load_data(
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        tokenizer: tr.PreTrainedTokenizer,
        shuffle: bool,
        process_sample_fn: Callable,
        max_positives: int,
        max_negatives: int,
        max_hard_negatives: int,
        max_contexts: int,
        max_question_length: int,
        max_context_length: int,
        *args,
        **kwargs,
    ) -> Any:
        # The data is a list of dictionaries, each dictionary is a sample
        # Each sample has the following keys:
        #   - "question": the question
        #   - "answers": a list of answers
        #   - "positive_ctxs": a list of positive contexts
        #   - "negative_ctxs": a list of negative contexts
        #   - "hard_negative_ctxs": a list of hard negative contexts
        # use the huggingface dataset library to load the data, by default it will load the
        # data in a dict with the key being "train". datasets needs str paths and not Path
        data = load_dataset("json", data_files=[str(p) for p in paths])["train"]
        data = data.map(
            partial(
                process_sample_fn,
                tokenizer=tokenizer,
                max_positives=max_positives,
                max_negatives=max_negatives,
                max_hard_negatives=max_hard_negatives,
                max_contexts=max_contexts,
                max_question_length=max_question_length,
                max_context_length=max_context_length,
            ),
            keep_in_memory=True,
            load_from_cache_file=True,
            num_proc=psutil.cpu_count(),
        )
        # shuffle the data
        if shuffle:
            data.shuffle(seed=42)
        # add id if not present
        data = data.add_column("sample_idx", range(len(data)))
        return data

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
        # remove duplicates and limit the number of contexts
        positive_ctxs = list(set([p["text"].strip() for p in sample["positive_ctxs"]]))
        if max_positives != -1:
            positive_ctxs = positive_ctxs[:max_positives]
        negative_ctxs = list(set([n["text"].strip() for n in sample["negative_ctxs"]]))
        if max_negatives != -1:
            negative_ctxs = negative_ctxs[:max_negatives]
        hard_negative_ctxs = list(
            set([h["text"].strip() for h in sample["hard_negative_ctxs"]])
        )
        if max_hard_negatives != -1:
            hard_negative_ctxs = hard_negative_ctxs[:max_hard_negatives]

        if "doc_topic" in sample:
            question = tokenizer(
                sample["question"],
                sample["doc_topic"],
                max_length=max_question_length,
                truncation=True,
            )
        else:
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

    def sample_dataset_negatives(self, seed: int = 42):
        """
        Wrapper around _sample_dataset_negatives to use it
        from external classes, like the callbacks.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to 42.
        """
        # seed numpy
        np.random.seed(seed)
        self.data = self._sample_dataset_negatives(
            self.data,
            self.tokenizer,
            self.context_manager,
            self.sample_by_frequency,
            self.max_negatives_to_sample,
            self.max_context_length,
        )

    def _sample_dataset_negatives(
        self,
        data,
        tokenizer: tr.PreTrainedTokenizer,
        context_manager: Labels,
        sample_by_frequency: bool = True,
        max_negatives_to_sample: int = 64,
        max_context_length: int = 64,
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
            context_frequencies = np.ones(context_manager.get_label_size())
            sampled_context_manager = context_manager
        sample_space_size = sampled_context_manager.get_label_size()
        logger.log(f"Sampling negative contexts from {sample_space_size} samples")
        # update the samples with the sampled negatives
        population = [i for i in range(sample_space_size)]
        data = data.map(
            partial(
                self._sample_negatives,
                tokenizer=tokenizer,
                population=population,
                context_frequencies=context_frequencies,
                context_manager=sampled_context_manager,
                max_negatives_to_sample=max_negatives_to_sample,
                max_context_length=max_context_length,
            ),
            keep_in_memory=True,
            num_proc=psutil.cpu_count(logical=False),
        )
        return data

    @staticmethod
    def _sample_negatives(
        sample: Dict[str, Any],
        tokenizer: tr.PreTrainedTokenizer,
        context_manager: Labels,
        population: List[int],
        context_frequencies: np.array,
        max_negatives_to_sample: int,
        max_context_length: int,
    ):
        """
        Sample negatives and add them to the sample.
        """

        positives_contexts_ids = sample["positive_ctxs"]
        negative_contexts_ids = sample["negative_ctxs"]
        hard_negative_contexts_ids = sample["hard_negative_ctxs"]
        # retrieved_hard_negatives = sample.get("retrieved_hard_negatives", [])

        positives = sample["positives"]
        positive_indices = [context_manager.get_index_from_label(p) for p in positives]
        # put to 0 the frequency of the positive contexts
        context_frequencies[positive_indices] = 0
        # normalize the frequencies
        context_frequencies = context_frequencies / np.sum(context_frequencies)

        actual_number_of_contexts = (
            len(positives_contexts_ids)
            + len(negative_contexts_ids)
            + len(hard_negative_contexts_ids)
            # + len(retrieved_hard_negatives)
        )

        sampled_negative_contexts = []
        if max_negatives_to_sample > 0:
            if actual_number_of_contexts < max_negatives_to_sample:
                remaining_contexts = max_negatives_to_sample - actual_number_of_contexts
                sampled = choice(
                    population, remaining_contexts, p=context_frequencies, replace=False
                )
                # sampled = choices(
                #     population,
                #     weights=context_frequencies,
                #     k=remaining_contexts,
                # )
                sampled_negative_contexts = [
                    context_manager.get_label_from_index(s) for s in sampled
                ]

        sampled_negative_ids = [
            tokenizer(n, max_length=max_context_length, truncation=True)
            for n in sampled_negative_contexts
        ]
        sample["sampled_negative_ctxs"] = sampled_negative_ids
        context = (
            positives_contexts_ids
            + negative_contexts_ids
            + hard_negative_contexts_ids
            + sampled_negative_ids
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


class DPRIterableDataset(GenerativeDataset, DPRMixin):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        shuffle: bool = False,
        max_contexts_per_batch: int = 32,
        max_questions_per_batch: int = 32,
        max_contexts: int = 64,
        max_positives: int = 1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 128,
        max_negatives_to_sample: int = 0,
        sample_by_frequency: bool = False,
        contexts_path: Union[str, os.PathLike] = None,
        tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        prefetch_batches: bool = False,
        **kwargs,
    ):
        super().__init__(
            name,
            path,
            max_tokens_per_batch=None,
            shuffle=shuffle,
            **kwargs,
        )
        self.max_contexts_per_batch = max_contexts_per_batch
        self.max_questions_per_batch = max_questions_per_batch
        self.max_contexts = max_contexts
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length
        self.max_negatives_to_sample = max_negatives_to_sample
        self.sample_by_frequency = sample_by_frequency
        self.prefetch_batches = prefetch_batches

        if type(self) == DPRDataset and max_positives != 1:
            raise ValueError(
                "DPRIterableDataset only supports one positive per question. "
                "Please use `InBatchNegativesDPRDataset` for multiple positives."
            )
        self.context_manager = Labels()
        # read contexts from file if provided
        if contexts_path:
            with open(self.project_folder / contexts_path, "r") as f:
                self.context_manager.add_labels(
                    [line.strip() for line in f.readlines()]
                )

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.padding_ops = {}
        else:
            if isinstance(self.tokenizer, str):
                self.tokenizer = tr.AutoTokenizer.from_pretrained(self.tokenizer)
            self.padding_ops = {
                "input_ids": partial(
                    self.pad_sequence,
                    value=self.tokenizer.pad_token_id,
                ),
                # value is None because: (read `pad_sequence` doc)
                "attention_mask": partial(self.pad_sequence, value=0),
                "token_type_ids": partial(
                    self.pad_sequence,
                    value=self.tokenizer.pad_token_type_id,
                ),
            }
        self.data = self.load(path, tokenizer=self.tokenizer, shuffle=shuffle)
        #        if self.max_negatives_to_sample > 0:
        #           self.data = self._sample_dataset_negatives(
        #               self.data,
        #              self.tokenizer,
        #             self.context_manager,
        #            self.sample_by_frequency,
        #           self.max_negatives_to_sample,
        #          self.max_context_length,
        #     )
        self.prefatched_data = []
        if self.prefetch_batches:
            self.prefetch()

    def batch_generator(self) -> List:
        batch = []
        contexts_in_batch = set()
        for sample in self.data:
            if len(contexts_in_batch) >= self.max_contexts_per_batch:
                yield batch
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
        if not self.drop_last_batch and len(batch) > 0:
            yield batch

    def collate_generator(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        collated_batch = self.collate_fn(batch)
        if (
            self.max_questions_per_batch
            and len(collated_batch.questions.input_ids) >= self.max_questions_per_batch
        ):
            splitted_batches = self.split_batch(
                collated_batch, self.max_questions_per_batch
            )
            for splitted_batch in splitted_batches:
                yield splitted_batch
        else:
            yield collated_batch

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.prefatched_data:
            for sample in self.prefatched_data:
                yield sample
            self.prefatched_data = []
            return
        else:
            for batch in self.batch_generator():
                for collated_batch in self.collate_generator(batch):
                    yield collated_batch
            return

    def prefetch(self):
        if self.prefetch_batches:
            self.prefatched_data = list(
                tqdm(self.batch_generator(), desc="Prefetching batches")
            )
            if self.max_negatives_to_sample > 0:
                # sample negatives for each batch
                self.prefatched_data = [
                    self._sample_batch_negatives(
                        batch,
                        self.tokenizer,
                        self.context_manager,
                        self.max_negatives_to_sample,
                        self.max_context_length,
                    )
                    for batch in tqdm(self.prefatched_data, desc="Sampling negatives")
                ]
            # collate batches
            collated_data = []
            for batch in tqdm(self.prefatched_data, desc="Collating batches"):
                collated_data.extend(self.collate_generator(batch))
            self.prefatched_data = collated_data

    def _sample_batch_negatives(
        self,
        batch: Dict[str, torch.Tensor],
        tokenizer: tr.PreTrainedTokenizer,
        context_manager: Labels,
        max_negatives_to_sample: int,
        max_context_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample negatives for each batch.
        """
        # sample negatives for each batch
        # batch["sampled_negative_ctxs"] = [
        #     self._sample_negatives(
        #         sample,
        #         tokenizer,
        #         context_manager,
        #         sample_by_frequency,
        #         max_negatives_to_sample,
        #         max_context_length,
        #     )
        #     for sample in batch["negative_ctxs"]
        # ]

        positives = set(
            [positive for sample in batch for positive in sample["positives"]]
        )
        positive_indices = [context_manager.get_index_from_label(p) for p in positives]

        context_frequencies = np.ones(self.context_manager.get_label_size())
        population = np.arange(self.context_manager.get_label_size())
        # put to 0 the frequency of the positive contexts
        context_frequencies[positive_indices] = 0
        # normalize the frequencies
        context_frequencies = context_frequencies / np.sum(context_frequencies)
        sampled = choice(
            population, max_negatives_to_sample, p=context_frequencies, replace=False
        )
        # sampled = choices(
        #     population,
        #     weights=context_frequencies,
        #     k=remaining_contexts,
        # )
        sampled_negative_contexts = [
            context_manager.get_label_from_index(s) for s in sampled
        ]
        sampled_negative_ids = [
            tokenizer(n, max_length=max_context_length, truncation=True)
            for n in sampled_negative_contexts
        ]
        # add the sampled negative contexts to each sample in the batch
        for sample in batch:
            sample["sampled_negative_ctxs"] = sampled_negative_ids
        return batch

    @staticmethod
    def split_batch(
        batch: Union[Dict[str, Any], ModelInputs], max_questions_per_batch: int
    ) -> List[ModelInputs]:
        """
        Split a batch into multiple batches of size `max_questions_per_batch` while keeping
        the same number of contexts.
        """
        # the batch should contain the following data:
        # {
        #     "questions": {
        #       "input_ids": torch.Tensor,
        #       "attention_mask": torch.Tensor,
        #       "token_type_ids": torch.Tensor,
        #     },
        #     "contexts": {
        #       "input_ids": torch.Tensor,
        #       "attention_mask": torch.Tensor,
        #       "token_type_ids": torch.Tensor,
        #     },
        #     "labels": torch.Tensor,
        #     "positives": Set,
        #     "sample_idx": List,
        # }
        # we want to split the questions, the labels and the sample_idx
        # we want to keep the same number of contexts and positives

        # split the questions
        questions = batch.questions
        questions = {
            key: torch.split(value, max_questions_per_batch, dim=0)
            for key, value in questions.items()
        }
        # split the labels
        labels = torch.split(batch.labels, max_questions_per_batch, dim=0)
        # reset the sample_idx
        sample_idx = batch.sample_idx
        sample_idx = [
            sample_idx[i : i + max_questions_per_batch]
            for i in range(0, len(sample_idx), max_questions_per_batch)
        ]
        # chunk the positives
        positives = batch.positives
        positives = [
            positives[i : i + max_questions_per_batch]
            for i in range(0, len(positives), max_questions_per_batch)
        ]
        # create the new batches
        new_batches = []
        for i, (q, l, s, p) in enumerate(
            zip(questions["input_ids"], labels, sample_idx, positives)
        ):
            new_batch = {
                "questions": ModelInputs(
                    {key: value[i] for key, value in questions.items()}
                ),
                "contexts": batch.contexts,
                "labels": l,
                "positives": p,
                "sample_idx": s,
            }
            new_batches.append(ModelInputs(new_batch))
        return new_batches

    def load(
        self,
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        tokenizer: tr.PreTrainedTokenizer = None,
        shuffle: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        if isinstance(paths, Sequence):
            paths = [self.project_folder / path for path in paths]
        else:
            paths = [self.project_folder / paths]

        # read the data and put it in a placeholder list
        for path in paths:
            if not path.exists():
                raise ValueError(f"{path} does not exist")

        # measure how long the preprocessing takes
        start = time.time()

        data = self.load_data(
            paths,
            tokenizer,
            shuffle,
            self._process_dataset_sample,
            self.max_positives,
            self.max_negatives,
            self.max_hard_negatives,
            self.max_contexts,
            self.max_question_length,
            self.max_context_length,
        )
        # convert to iterable dataset
        # data = data.to_iterable_dataset(num_shards=4)

        end = time.time()
        logger.log(
            f"Pre-processing [bold cyan]{self.name}[/bold cyan] "
            f"data took [bold]{end - start:.2f}[/bold] seconds"
        )
        return data

    def save_data(
        self, path: Union[str, os.PathLike], remove_columns: Optional[List[str]] = None
    ) -> None:
        """
        Save the samples to a file.

        Args:
            path (:obj:`str`):
                Path to the file where to save the dataset.
            remove_columns (:obj:`str`):
                Data not to save on disk
        """
        if remove_columns is None:
            remove_columns = []
        with open(path, "w") as f:
            for sample in self.data:
                sample["question"] = self.tokenizer.decode(
                    sample["question"]["input_ids"]
                )
                # remove columns if needed
                for key in remove_columns:
                    if key in sample:
                        sample.pop(key)
                json.dump(sample, f, indent=2)

    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        questions = [sample["question"] for sample in batch]
        contexts = [sample["context"] for sample in batch]
        positives = [sample["positives"] for sample in batch]

        questions = self.convert_to_batch(questions)
        # first flat the list of list of contexts
        contexts = [c for ctxs in contexts for c in ctxs]
        # invert contexts from list of dict to dict of list
        contexts = self.convert_to_batch(contexts)

        # actual positives
        labels = torch.zeros(
            questions["input_ids"].shape[0], contexts["input_ids"].shape[0]
        )
        positive_index_end = [sample["positive_index_end"] for sample in batch]
        last_start = 0
        for i, end in enumerate(positive_index_end):
            start = 0 if i == 0 else last_start + len(batch[i - 1]["context"])
            end = end if i == 0 else start + end
            labels[i, start:end] = 1
            last_start = start

        model_inputs = {
            "questions": ModelInputs(questions),
            "contexts": ModelInputs(contexts),
            "labels": labels,
            "positives": positives,
            "sample_idx": [sample["sample_idx"] for sample in batch],
        }
        return ModelInputs(model_inputs)


class InBatchNegativesDPRIterableDataset(DPRIterableDataset, DPRMixin):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        shuffle: bool = False,
        max_contexts_per_batch: int = 64,
        max_questions_per_batch: int = 64,
        max_contexts: int = -1,
        max_positives: int = -1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 128,
        max_negatives_to_sample: int = 0,
        sample_by_frequency: bool = False,
        contexts_path: Union[str, os.PathLike] = None,
        tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            path,
            shuffle,
            max_contexts_per_batch,
            max_questions_per_batch,
            max_contexts,
            max_positives,
            max_negatives,
            max_hard_negatives,
            max_question_length,
            max_context_length,
            max_negatives_to_sample,
            sample_by_frequency,
            contexts_path,
            tokenizer,
            **kwargs,
        )

    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        # get data from batch
        questions = [sample["question"] for sample in batch]
        positives = [sample["positives"] for sample in batch]

        # this is needed to get the correct labels for each question
        positives_ctxs = [sample["positive_ctxs"] for sample in batch]
        negatives_ctxs = [sample["negative_ctxs"] for sample in batch]
        if "sampled_negative_ctxs" in batch[0]:
            negatives_ctxs += [sample["sampled_negative_ctxs"] for sample in batch]

        hard_negatives_ctxs = [sample["hard_negative_ctxs"] for sample in batch]
        # use negatives from predictions if available
        if "retrieved_hard_negatives" in batch[0]:
            # add augmented negative contexts to contexts
            hard_negatives_ctxs += [
                sample["retrieved_hard_negatives"] for sample in batch
            ]

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
            "questions": ModelInputs(questions),
            "contexts": ModelInputs(contexts),
            "labels": labels,
            "positives": positives,
            "sample_idx": [sample["sample_idx"] for sample in batch],
        }
        return ModelInputs(model_inputs)


class DPRDataset(BaseDataset, DPRMixin):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        shuffle: bool = False,
        max_contexts: int = 64,
        max_positives: int = 1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 128,
        max_negatives_to_sample: int = 0,
        sample_by_frequency: bool = False,
        contexts_path: Union[str, os.PathLike] = None,
        tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)
        self.max_contexts = max_contexts
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length
        self.max_negatives_to_sample = max_negatives_to_sample
        self.sample_by_frequency = sample_by_frequency

        if type(self) == DPRDataset and max_positives != 1:
            raise ValueError(
                "DPRDataset only supports one positive per question. "
                "Please use `InBatchNegativesDPRDataset` or `SampledNegativesDPRDataset` "
                "for multiple positives."
            )
        self.context_manager = Labels()
        # read contexts from file if provided
        if contexts_path:
            with open(self.project_folder / contexts_path, "r") as f:
                self.context_manager.add_labels(
                    [line.strip() for line in f.readlines()]
                )

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.padding_ops = {}
        else:
            if isinstance(self.tokenizer, str):
                self.tokenizer = tr.AutoTokenizer.from_pretrained(self.tokenizer)
            self.padding_ops = {
                "input_ids": partial(
                    self.pad_sequence,
                    value=self.tokenizer.pad_token_id,
                ),
                # value is None because: (read `pad_sequence` doc)
                "attention_mask": partial(self.pad_sequence, value=0),
                "token_type_ids": partial(
                    self.pad_sequence,
                    value=self.tokenizer.pad_token_type_id,
                ),
            }
        self.data = self.load(path, tokenizer=self.tokenizer, shuffle=shuffle)
        if self.max_negatives_to_sample > 0:
            self.data = self._sample_dataset_negatives(
                self.data,
                self.tokenizer,
                self.context_manager,
                self.sample_by_frequency,
                self.max_negatives_to_sample,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.data[index]

    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        questions = [sample["question"] for sample in batch]
        contexts = [sample["context"] for sample in batch]
        positives = [sample["positives"] for sample in batch]

        questions = self.convert_to_batch(questions)
        # first flat the list of list of contexts
        contexts = [c for ctxs in contexts for c in ctxs]
        # invert contexts from list of dict to dict of list
        contexts = self.convert_to_batch(contexts)

        # actual positives
        labels = torch.zeros(
            questions["input_ids"].shape[0], contexts["input_ids"].shape[0]
        )
        positive_index_end = [sample["positive_index_end"] for sample in batch]
        last_start = 0
        for i, end in enumerate(positive_index_end):
            start = 0 if i == 0 else last_start + len(batch[i - 1]["context"])
            end = end if i == 0 else start + end
            labels[i, start:end] = 1
            last_start = start

        model_inputs = {
            "questions": ModelInputs(questions),
            "contexts": ModelInputs(contexts),
            "labels": labels,
            "positives": positives,
            "sample_idx": [sample["sample_idx"] for sample in batch],
        }
        return ModelInputs(model_inputs)

    def load(
        self,
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        tokenizer: tr.PreTrainedTokenizer = None,
        shuffle: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        if isinstance(paths, Sequence):
            paths = [self.project_folder / path for path in paths]
        else:
            paths = [self.project_folder / paths]

        # read the data and put it in a placeholder list
        for path in paths:
            if not path.exists():
                raise ValueError(f"{path} does not exist")

        # measure how long the preprocessing takes
        start = time.time()

        data = self.load_data(
            paths,
            tokenizer,
            shuffle,
            self._process_dataset_sample,
            self.max_positives,
            self.max_negatives,
            self.max_hard_negatives,
            self.max_contexts,
            self.max_question_length,
            self.max_context_length,
        )

        end = time.time()
        logger.log(
            f"Pre-processing [bold cyan]{self.name}[/bold cyan] "
            f"data took [bold]{end - start:.2f}[/bold] seconds"
        )
        return data


class InBatchNegativesDPRDataset(DPRDataset):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        shuffle: bool = False,
        max_contexts: int = -1,
        max_positives: int = -1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 128,
        max_negatives_to_sample: int = 0,
        sample_by_frequency: bool = False,
        contexts_path: Union[str, os.PathLike] = None,
        tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            path,
            shuffle,
            max_contexts,
            max_positives,
            max_negatives,
            max_hard_negatives,
            max_question_length,
            max_context_length,
            max_negatives_to_sample,
            sample_by_frequency,
            contexts_path,
            tokenizer,
            **kwargs,
        )

    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        # get data from batch
        questions = [sample["question"] for sample in batch]
        positives = [sample["positives"] for sample in batch]

        # this is needed to get the correct labels for each question
        positives_ctxs = [sample["positive_ctxs"] for sample in batch]
        negatives_ctxs = [sample["negative_ctxs"] for sample in batch]
        hard_negatives_ctxs = [sample["hard_negative_ctxs"] for sample in batch]
        # use negatives from predictions if available
        if "retrieved_hard_negatives" in batch[0]:
            # add augmented negative contexts to contexts
            hard_negatives_ctxs += [
                sample["retrieved_hard_negatives"] for sample in batch
            ]

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
            "questions": ModelInputs(questions),
            "contexts": ModelInputs(contexts),
            "labels": labels,
            "positives": positives,
            "sample_idx": [sample["sample_idx"] for sample in batch],
        }
        return ModelInputs(model_inputs)


class SampledNegativesDPRDataset(DPRDataset):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        shuffle: bool = False,
        max_contexts: int = 64,
        max_positives: int = -1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 128,
        max_negatives_to_sample: int = 64,
        sample_by_frequency: bool = False,
        contexts_path: Union[str, os.PathLike] = None,
        tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            path,
            shuffle,
            max_contexts,
            max_positives,
            max_negatives,
            max_hard_negatives,
            max_question_length,
            max_context_length,
            max_negatives_to_sample,
            sample_by_frequency,
            contexts_path,
            tokenizer,
        )

    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        questions = [sample["question"] for sample in batch]
        contexts = [sample["context"] for sample in batch]
        positives = [sample["positives"] for sample in batch]
        if "retrieved_hard_negatives" in batch[0]:
            # add augmented negative contexts to contexts
            retrieved_hard_negatives = [
                sample["retrieved_hard_negatives"] for sample in batch
            ]
            contexts = [
                # remove the last len(a) contexts to add the augmented negative context
                c[: -len(a)] + a
                for c, a in zip(contexts, retrieved_hard_negatives)
            ]

        questions = self.convert_to_batch(questions)
        # first flat the list of lists of contexts
        contexts = [c for ctxs in contexts for c in ctxs]
        # invert contexts from list of dict to dict of list
        contexts = self.convert_to_batch(contexts)

        augmented_labels: Optional[torch.Tensor] = None
        contexts_per_question = [len(sample["context"]) for sample in batch]
        labels = [[0] * c for c in contexts_per_question]
        # pad the labels
        labels = [
            self.pad_sequence(l, max(contexts_per_question), value=-100) for l in labels
        ]
        # convert to tensor
        labels = torch.as_tensor(labels)
        # labels is a mask of positive contexts for each question base on positive_index_end
        # has shape num_questions x num_contexts
        positive_index_end = [sample["positive_index_end"] for sample in batch]
        for i, end in enumerate(positive_index_end):
            labels[i, :end] = 1

        model_inputs = {
            "questions": ModelInputs(questions),
            "contexts": ModelInputs(contexts),
            "labels": augmented_labels if augmented_labels is not None else labels,
            "positives": positives,
            "sample_idx": [sample["sample_idx"] for sample in batch],
        }
        if contexts_per_question is not None:
            model_inputs["contexts_per_question"] = contexts_per_question
        return ModelInputs(model_inputs)
