import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union, Optional

import numpy as np
import psutil
import torch
import transformers as tr
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset

from data.labels import Labels
from utils.logging import get_console_logger
from utils.model_inputs import ModelInputs

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
        max_tokens_per_batch: int = 800,
        drop_last_batch: bool = False,
        shuffle: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.max_tokens_per_batch = max_tokens_per_batch
        self.drop_last_batch = drop_last_batch
        self.shuffle = shuffle
        self.data = self.load(path)
        self.n_batches = sum([1 for _ in self])

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


class DPRDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        shuffle: bool = False,
        pre_process: bool = True,
        max_contexts: int = 64,
        max_positives: int = -1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 128,
        shuffle_negative_contexts: bool = False,
        in_batch_positives_augmentation: bool = True,
        tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        contexts_path: Union[str, os.PathLike] = None,
        strategy: str = "fixed_contexts",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)
        self.max_contexts = max_contexts
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length
        self.shuffle_negative_contexts = shuffle_negative_contexts
        self.in_batch_positives_augmentation = in_batch_positives_augmentation
        self.strategy = strategy

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

        self.data = self.load(
            path, tokenizer=self.tokenizer, pre_process=pre_process, shuffle=shuffle
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.data[index]

    def load(
        self,
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        tokenizer: tr.PreTrainedTokenizer = None,
        pre_process: bool = True,
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

        if pre_process:
            if not tokenizer:
                raise ValueError("Tokenizer is required for pre-processing")
            # Pre-process the data
            if shuffle:
                # shuffle the data
                data = data.shuffle(seed=42)

            # measure how long the preprocessing takes
            start = time.time()
            data = data.map(
                partial(
                    DPRDataset.process_sample,
                    tokenizer=tokenizer,
                    max_contexts=self.max_contexts,
                    max_positives=self.max_positives,
                    max_negatives=self.max_negatives,
                    max_hard_negatives=self.max_hard_negatives,
                    context_manager=self.context_manager,
                    strategy=self.strategy,
                ),
                remove_columns=[
                    "answers",
                    "positive_ctxs",
                    "negative_ctxs",
                    "hard_negative_ctxs",
                ],
                keep_in_memory=True,
                load_from_cache_file=True,
                num_proc=psutil.cpu_count(),
            )
            # add id if not present
            if "id" not in data.column_names:
                data = data.add_column("id", range(len(data)))
            end = time.time()
            logger.log(
                f"Pre-processing [bold cyan]{self.name}[/bold cyan] "
                f"data took [bold]{end - start:.2f}[/bold] seconds"
            )
        return data

    @property
    def contexts(self):
        return self.context_manager.get_labels()

    @staticmethod
    def process_sample(
        sample: Dict,
        tokenizer: tr.PreTrainedTokenizer,
        max_positives: int,
        max_negatives: int,
        max_hard_negatives: int,
        max_contexts: int = -1,
        max_question_length: int = 256,
        max_context_length: int = 128,
        context_manager: Labels = None,
        strategy: str = "fixed_contexts",
    ) -> Dict:
        positive_ctxs = [p["text"] for p in sample["positive_ctxs"]]
        if max_positives != -1:
            positive_ctxs = positive_ctxs[:max_positives]
        negative_ctxs = [n["text"] for n in sample["negative_ctxs"]]
        if max_negatives != -1:
            negative_ctxs = negative_ctxs[:max_negatives]
        hard_negative_ctxs = [h["text"] for h in sample["hard_negative_ctxs"]]
        if max_hard_negatives != -1:
            hard_negative_ctxs = hard_negative_ctxs[:max_hard_negatives]

        if strategy == "fixed_contexts":
            positive_indices = [
                context_manager.get_index_from_label(p) for p in positive_ctxs
            ]
            number_of_positives = len(positive_ctxs)
            actual_number_of_contexts = (
                number_of_positives + len(negative_ctxs) + len(hard_negative_ctxs)
            )
            if max_contexts != -1:
                if actual_number_of_contexts < max_contexts:
                    remaining_contexts = max_contexts - actual_number_of_contexts

                    sampled = DPRDataset.fast_sampling(
                        context_manager.get_label_size(),
                        remaining_contexts,
                        exclude=positive_indices,
                    )

                    negative_ctxs += [
                        context_manager.get_label_from_index(s) for s in sampled[0]
                    ]

                else:
                    # TODO: limit the number of contexts to max_contexts (shouldn't happen)
                    pass

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
        output = {
            "question": question,
            "context": context,
            "positives": set([p["text"] for p in sample["positive_ctxs"]]),
            "positive_index_end": len(positive_ctxs),
        }
        return output

    @staticmethod
    def fast_sampling(
        num_elements: int,
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
            num_elements (`int`): number of elements to sample from
            sample_size (`int`): number of elements to sample
            num_samples (`int`, optional): number of samples to draw. Defaults to 1.
            probabilities (`np.array`, optional): probabilities of each element. Defaults to None.
            exclude (`List[int]`, optional): indices of elements to exclude. Defaults to None.

        Returns:
            `np.array`: array of sampled indices
        """
        if probabilities is None:
            # probabilities should sum to 1
            probabilities = np.random.random(num_elements)
            probabilities /= np.sum(probabilities)

        if exclude is not None:
            probabilities[exclude] = 0
            probabilities /= np.sum(probabilities)

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

    # @staticmethod
    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        questions = [sample["question"] for sample in batch]
        contexts = [sample["context"] for sample in batch]
        positives = [sample["positives"] for sample in batch]
        if "augmented_negative_contexts" in batch[0]:
            # add augmented negative contexts to contexts
            augmented_negative_contexts = [
                sample["augmented_negative_contexts"] for sample in batch
            ]
            contexts = [c + a for c, a in zip(contexts, augmented_negative_contexts)]

        questions = self.convert_to_batch(questions)
        # first flat the list of lists of contexts
        contexts = [c for ctxs in contexts for c in ctxs]
        # invert contexts from list of dict to dict of list
        contexts = self.convert_to_batch(contexts)

        # actual positives
        labels = torch.zeros(
            questions["input_ids"].shape[0], contexts["input_ids"].shape[0]
        )
        augmented_labels: Optional[torch.Tensor] = None
        if self.strategy == "in_batch_negatives":
            positive_index_end = [sample["positive_index_end"] for sample in batch]
            last_start = 0
            for i, end in enumerate(positive_index_end):
                start = 0 if i == 0 else last_start + len(batch[i - 1]["context"])
                end = end if i == 0 else start + end
                labels[i, start:end] = 1
                last_start = start

            if self.in_batch_positives_augmentation:
                # labels is a mask of positive contexts for each question
                # has shape num_questions x num_contexts
                augmented_labels = torch.zeros(
                    questions["input_ids"].shape[0], contexts["input_ids"].shape[0]
                )
                flat_positives = [s for sample in batch for s in sample["positives"]]
                # labels includes as positive also the labels that appear in the
                # other samples but are positive for the considered one (avoid false
                # negative context)
                for p_idx, p in enumerate(flat_positives):
                    for i, positive in enumerate(positives):
                        for positive_ctx in positive:
                            if positive_ctx in p:
                                augmented_labels[i, p_idx] = 1
        else:
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
            "id": [sample["id"] for sample in batch],
        }
        return ModelInputs(model_inputs)

    def save_data(self, path: Union[str, os.PathLike]) -> None:
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
        # too large to save
        samples.pop("context")

        with open(path, "w") as f:
            json.dump(samples, f)

    def update_data(
        self,
        names: Union[str, List[str]],
        columns: Union[List[Any], List[List[Any]]],
        overwrite: bool = False,
    ):
        """
        Update the data with the given column.

        Args:
            names (:obj:`str` or :obj:`List[str]`):
                Name of the column to update.
            columns (:obj:`List[Any]` or :obj:`List[List[Any]]`):
                List of values to update.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to overwrite the column if it already exists.
        """
        # TODO: augment data if overwrite is False and the column already exists
        if isinstance(names, str):
            names = [names]
            # TODO: check if it's a list of list in a prettier way
            if len(names) != len(columns):
                columns = [columns]

        if len(names) != len(columns):
            raise ValueError(
                "The number of columns to update must be equal to the number of names."
            )

        for name, column in zip(names, columns):
            if name in self.data.column_names:
                if not overwrite:
                    raise ValueError(
                        "The dataset already contains a column named `name`, you can force the update by "
                        "setting `overwrite=True`."
                    )
                self.data = self.data.remove_columns(name)
            self.data = self.data.add_column(name=name, column=column)
