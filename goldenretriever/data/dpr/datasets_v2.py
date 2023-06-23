from enum import Enum
import json
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import psutil
import torch
import transformers as tr
from datasets import IterableDataset, load_dataset
from numpy.random import choice
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from goldenretriever.common.log import get_console_logger, get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.dpr.hard_negatives_manager import HardNegativeManager
from goldenretriever.data.dpr.negatives import InBatchNegativesStrategy
from goldenretriever.data.labels import Labels

console_logger = get_console_logger()

logger = get_logger()


class SubsampleStrategyEnum(Enum):
    NONE = "none"
    RANDOM = "random"
    IN_ORDER = "in_order"


class InBatchNegativesDataset(Dataset):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]] = None,
        data: Any = None,
        context_batch_size: int = 32,
        question_batch_size: int = 32,
        max_positives: int = -1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 64,
        contexts_path: Union[str, os.PathLike] = None,
        tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        shuffle: bool = False,
        datasets_batch_size: int = 1000,
        subsample_strategy: Optional[str] = SubsampleStrategyEnum.NONE,
        subsample_portion: float = 0.1,
        load_from_cache_file: bool = True,
        num_proc: Optional[int] = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        **kwargs,
    ):
        super().__init__()

        if path is None and data is None:
            raise ValueError("Either `path` or `data` must be provided")

        if tokenizer is None:
            raise ValueError("A tokenizer must be provided")

        # dataset parameters
        self.name = name
        self.path = path
        self.project_folder = Path(__file__).parent.parent.parent.parent

        # dataset hyperparameters
        self.context_batch_size = context_batch_size
        self.question_batch_size = question_batch_size
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length
        self.shuffle = shuffle
        self.datasets_batch_size = datasets_batch_size

        # create a manager for the contexts
        self.context_manager = Labels()
        # read contexts from file if provided
        if contexts_path:
            with open(self.project_folder / contexts_path, "r") as f:
                self.context_manager.add_labels(
                    [line.strip() for line in f.readlines()]
                )

        # context_batch_size cannot be greater than the number of contexts
        if self.context_batch_size > self.context_manager.get_label_size():
            logger.info(
                f"Your context_batch_size ({context_batch_size}) "
                f"is greater than the number of contexts ({self.context_manager.get_label_size()}). "
                f"Setting context_batch_size to {self.context_manager.get_label_size()}."
            )
            self.context_batch_size = self.context_manager.get_label_size()

        self.tokenizer = tokenizer
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

        # initialize the Hard Negatives manager
        self.hard_negatives_manager: Optional[HardNegativeManager] = None

        # check if subsample strategy is valid
        if subsample_strategy is not None:
            # subsample_strategy can be a string or a SubsampleStrategy
            if isinstance(subsample_strategy, str):
                try:
                    subsample_strategy = SubsampleStrategyEnum(subsample_strategy)
                except ValueError:
                    raise ValueError(
                        f"Subsample strategy {subsample_strategy} is not valid. "
                        f"Valid strategies are: {SubsampleStrategyEnum.__members__}"
                    )
            if not isinstance(subsample_strategy, SubsampleStrategyEnum):
                raise ValueError(
                    f"Subsample strategy {subsample_strategy} is not valid. "
                    f"Valid strategies are: {SubsampleStrategyEnum.__members__}"
                )

        self.subsample_strategy = subsample_strategy
        self.subsample_portion = subsample_portion

        # load the dataset
        if data is None:
            self.data: Dataset = self.load(
                path,
                tokenizer=self.tokenizer,
                load_from_cache_file=load_from_cache_file,
                num_proc=num_proc,
                shuffle=shuffle,
                keep_in_memory=keep_in_memory,
                streaming=streaming,
                max_positives=max_positives,
                max_negatives=max_negatives,
                max_hard_negatives=max_hard_negatives,
                max_question_length=max_question_length,
                max_context_length=max_context_length,
                **kwargs,
            )
        else:
            self.data: Dataset = data

        # keep track of how many times the dataset has been iterated over
        self.number_of_complete_iterations = 0

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.data[index]

    def __repr__(self) -> str:
        return f"Dataset({self.name=}, {self.path=})"

    def to_torch_dataset(self) -> torch.utils.data.Dataset:
        shuffle_this_time = self.shuffle

        if (
            self.subsample_strategy
            and self.subsample_strategy != SubsampleStrategyEnum.NONE
        ):
            if self.subsample_strategy == SubsampleStrategyEnum.RANDOM:
                logger.info(
                    f"Random subsampling {number_of_samples} samples from {len(self.data)}"
                )
                data = (
                    deepcopy(self.data)
                    .shuffle(seed=42 + self.number_of_complete_iterations)
                    .select(range(0, number_of_samples))
                )
            elif self.subsample_strategy == SubsampleStrategyEnum.IN_ORDER:
                number_of_samples = int(len(self.data) * self.subsample_portion)
                already_selected = (
                    number_of_samples * self.number_of_complete_iterations
                )
                logger.info(
                    f"Subsampling {number_of_samples} samples from {len(self.data)}"
                )
                to_select = min(already_selected + number_of_samples, len(self.data))
                logger.info(
                    f"Portion of data selected: {already_selected} "
                    f"to {already_selected + number_of_samples}"
                )
                data = deepcopy(self.data).select(range(already_selected, to_select))

                # don't shuffle the data if we are subsampling and we have still not completed
                # one full iteration over the dataset
                if self.number_of_complete_iterations > 0:
                    shuffle_this_time = False

                # reset the number of complete iterations
                if to_select >= len(self.data):
                    self.number_of_complete_iterations = 0
            else:
                raise ValueError(
                    f"Subsample strategy {self.subsample_strategy} is not valid. "
                    f"Valid strategies are: {SubsampleStrategyEnum.__members__}"
                )

        else:
            data = data = self.data

        # do we need to shuffle the data?
        if self.shuffle and shuffle_this_time:
            self.shuffle_data(seed=42 + self.number_of_complete_iterations)

        batched_data = self.create_batches(
            data,
            batch_fn=self.batch_fn,
            datasets_batch_size=self.datasets_batch_size,
            context_batch_size=self.context_batch_size,
            question_batch_size=self.question_batch_size,
            hard_negatives_manager=self.hard_negatives_manager,
        )

        batched_data = self.collate_batches(batched_data, self.collate_fn)

        # increment the number of complete iterations
        self.number_of_complete_iterations += 1

        return batched_data.with_format("torch")

    @staticmethod
    def create_batches(
        data: Dataset,
        batch_fn: Callable,
        datasets_batch_size: int,
        context_batch_size: int,
        question_batch_size: int,
        hard_negatives_manager: Optional[HardNegativeManager] = None,
        num_proc: Optional[int] = None,
    ) -> Dataset:
        if num_proc is None:
            num_proc = psutil.cpu_count(logical=False)

        batched_data = data.map(
            batch_fn,
            fn_kwargs=dict(
                hard_negatives_manager=hard_negatives_manager,
                context_batch_size=context_batch_size,
                question_batch_size=question_batch_size,
            ),
            batched=True,
            batch_size=datasets_batch_size,
            remove_columns=data.column_names,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc="Creating batches",
        )
        return batched_data

    @staticmethod
    def collate_batches(
        data: Dataset,
        collate_fn: Callable,
        num_proc: Optional[int] = None,
    ) -> Dataset:
        if num_proc is None:
            num_proc = psutil.cpu_count(logical=False)

        collated_data = data.map(
            collate_fn,
            remove_columns=data.column_names,
            num_proc=1,
            load_from_cache_file=False,
            desc="Collating batches",
        )
        return collated_data

    @staticmethod
    def _map_tokenize_sample(
        sample: Dict,
        tokenizer: tr.PreTrainedTokenizer,
        max_positives: int,
        max_negatives: int,
        max_hard_negatives: int,
        max_contexts: int = -1,
        max_question_length: int = 256,
        max_context_length: int = 128,
    ) -> Dict:
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
        output = dict(
            question=question,
            context=context,
            positives=set([p["text"].strip() for p in sample["positive_ctxs"]]),
            positive_ctxs=positive_ctxs,
            negative_ctxs=negative_ctxs,
            hard_negative_ctxs=hard_negative_ctxs,
            positive_index_end=len(positive_ctxs),
        )
        return output

    @staticmethod
    def batch_fn(
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
                # if hard_negatives_manager is not None:
                #     contexts_in_batch |= set(
                #         tuple(s["input_ids"])
                #         for sample in batch
                #         if sample["sample_idx"] in hard_negatives_manager
                #         for s in hard_negatives_manager.get(sample["sample_idx"])
                #     )
        if len(batch) > 0:
            output_batches["batches"].append(batch)

        return output_batches

    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
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

        if self.hard_negatives_manager is not None:
            for sample_idx in sample_idxs:
                hard_negatives_ctxs.append(self.hard_negatives_manager.get(sample_idx))

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

    def load(
        self,
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        tokenizer: tr.PreTrainedTokenizer = None,
        load_fn_kwargs: Dict = None,
        load_from_cache_file: bool = True,
        num_proc: Optional[int] = None,
        shuffle: bool = False,
        keep_in_memory: bool = True,
        streaming: bool = False,
        max_positives: int = -1,
        max_negatives: int = -1,
        max_hard_negatives: int = -1,
        max_contexts: int = -1,
        max_question_length: int = 256,
        max_context_length: int = 64,
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

        fn_kwargs = dict(
            tokenizer=tokenizer,
            max_positives=max_positives,
            max_negatives=max_negatives,
            max_hard_negatives=max_hard_negatives,
            max_contexts=max_contexts,
            max_question_length=max_question_length,
            max_context_length=max_context_length,
        )
        if load_fn_kwargs is not None:
            fn_kwargs.update(load_fn_kwargs)

        if num_proc is None:
            num_proc = psutil.cpu_count(logical=False)

        map_kwargs = dict(
            function=self._map_tokenize_sample,
            fn_kwargs=fn_kwargs,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )

        # The data is a list of dictionaries, each dictionary is a sample
        # Each sample has the following keys:
        #   - "question": the question
        #   - "answers": a list of answers
        #   - "positive_ctxs": a list of positive contexts
        #   - "negative_ctxs": a list of negative contexts
        #   - "hard_negative_ctxs": a list of hard negative contexts
        # use the huggingface dataset library to load the data, by default it will load the
        # data in a dict with the key being "train".
        logger.info("Loading data from files")
        data = load_dataset(
            "json",
            data_files=[str(p) for p in paths],  # datasets needs str paths and not Path
            split="train",
            streaming=streaming,
        )
        # add id if not present
        data = data.add_column("sample_idx", range(len(data)))

        # preprocess the data
        data = data.map(**map_kwargs)

        # shuffle the data
        if shuffle:
            data.shuffle(seed=42)

        return data

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

    def shuffle_data(self, seed: int = 42):
        self.data = self.data.shuffle(seed=seed)

    @property
    def contexts(self):
        return list(self.context_manager.get_labels().keys())
