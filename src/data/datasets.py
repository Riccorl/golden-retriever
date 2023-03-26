import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union, Optional, Callable

import psutil
import torch
import transformers as tr
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset

from data.dpr_mixin import DPRMixin
from data.labels import Labels
from common.logging import get_console_logger
from common.model_inputs import ModelInputs

logger = get_console_logger()


class BaseDataset(Dataset):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]] = None,
        data: Any = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        if path is None and data is None:
            raise ValueError("Either `path` or `data` must be provided")
        self.path = path
        self.project_folder = Path(__file__).parent.parent.parent
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.data[index]

    def __repr__(self) -> str:
        return f"Dataset({self.name=}, {self.path=})"

    def load(
        self,
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        *args,
        **kwargs,
    ) -> Any:
        # load data from single or multiple paths in one single dataset
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: Any, *args, **kwargs) -> Any:
        raise NotImplementedError


class GenerativeDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        path: Union[str, Path, List[str], List[Path]],
        max_tokens_per_batch: Optional[int] = 800,
        drop_last_batch: bool = False,
        shuffle: bool = False,
        data: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.max_tokens_per_batch = max_tokens_per_batch
        self.drop_last_batch = drop_last_batch
        self.shuffle = shuffle
        self.project_folder = Path(__file__).parent.parent.parent
        self.data = data
        # self.data = self.load(path)
        # self.n_batches = sum([1 for _ in self])

    def __repr__(self) -> str:
        return f"Dataset({self.name=}, {self.path=})"

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        batch = []
        ct = 0
        for sample in self.data:
            # number of tokens in the sample
            sample_tokens = len(sample)
            if (
                max(ct, sample_tokens) * (len(batch) + 1) > self.max_tokens_per_batch
                and len(batch) > 0
            ):
                yield self.prepare_output_batch(batch)
                batch = []
                ct = 0
            batch.append(sample)
            ct = max(ct, sample_tokens)
        # drop last cause might be too short and result in issues (nan if we are using amp)
        if not self.drop_last_batch and len(batch) > 0:
            yield self.prepare_output_batch(batch)

    def prepare_output_batch(self, batch: Any) -> Any:
        # Use this as `collate_fn`
        raise NotImplementedError

    def load(self, paths: Union[str, Path, List[str], List[Path]]) -> Any:
        # load data from single or multiple paths in one single dataset
        # it may be useful to shuffle the dataset here if this is a train dataset:
        # if self.shuffle:
        #   random.shuffle(data)
        raise NotImplementedError


class DPRLoadMixin:
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


class DPRIterableDataset(GenerativeDataset, DPRMixin, DPRLoadMixin):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        shuffle: bool = False,
        max_contexts_per_batch: int = 32,
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
        super().__init__(
            name,
            path,
            max_tokens_per_batch=None,
            shuffle=shuffle,
            **kwargs,
        )
        self.max_contexts_per_batch = max_contexts_per_batch
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
        if self.max_negatives_to_sample > 0:
            self.data = self._sample_dataset_negatives(
                self.data,
                self.tokenizer,
                self.context_manager,
                sample_by_frequency,
                self.max_negatives_to_sample,
            )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        batch = []
        contexts_in_batch = set()
        for sample in self.data:
            if len(contexts_in_batch) >= self.max_contexts_per_batch:
                yield self.collate_fn(batch)
                batch = []
                contexts_in_batch = set()
            batch.append(sample)
            for context_type in {
                "positive_ctxs",
                "negative_ctxs",
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
        # drop last cause might be too short and result in issues (nan if we are using amp)
        if not self.drop_last_batch and len(batch) > 0:
            yield self.collate_fn(batch)

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
        # convert to iterable dataset
        data = data.to_iterable_dataset(num_shards=4)

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


class InBatchNegativesDPRIterableDataset(DPRIterableDataset):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        shuffle: bool = False,
        max_contexts_per_batch: int = 64,
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
                sample_by_frequency,
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

    def save_data(
        self, path: Union[str, os.PathLike], remove_columns: Optional[List[str]] = None
    ) -> None:
        """
        Save the samples to a file.

        Args:
            path (:obj:`str`):
                Path to the file where to save the dataset.
        """
        samples = self.data.to_dict()
        samples["question"] = self.tokenizer.batch_decode(
            [question["input_ids"] for question in samples["question"]]
        )

        # remove columns if needed
        if remove_columns is None:
            remove_columns = []
        for key in remove_columns:
            if key in samples:
                samples.pop(key)

        with open(path, "w") as f:
            for i in range(len(samples["question"])):
                dump = {
                    k: v[i] if isinstance(v, list) else v for k, v in samples.items()
                }
                json.dump(dump, f, indent=2)


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
        sample_by_frequency: bool = True,
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
