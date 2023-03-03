import json
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union, Optional
import psutil

import torch
import transformers as tr
from rich.progress import track
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

from utils.logging import get_console_logger
from utils.model_inputs import ModelInputs

logger = get_console_logger()


class BaseDataset(Dataset):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]],
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.project_folder = Path(__file__).parent.parent.parent

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

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
        split_contexts: bool = False,
        max_positives: int = -1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 128,
        shuffle_negative_contexts: bool = False,
        in_batch_positives_augmentation: bool = True,
        tokenizer: tr.PreTrainedTokenizer = None,
        contexts_path: Union[str, os.PathLike] = None,
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)
        self.split_contexts = split_contexts
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length
        self.shuffle_negative_contexts = shuffle_negative_contexts
        self.in_batch_positives_augmentation = in_batch_positives_augmentation
        self.contexts: Optional[List[str]] = None
        # read contexts from file if provided
        if contexts_path:
            with open(self.project_folder / contexts_path, "r") as f:
                self.contexts = [line.strip() for line in f.readlines()]

        self.padding_ops = {
            "input_ids": partial(
                self.pad_sequence,
                value=tokenizer.pad_token_id,
            ),
            # value is None because: (read `pad_sequence` doc)
            "attention_mask": partial(self.pad_sequence, value=0),
            "token_type_ids": partial(
                self.pad_sequence,
                value=tokenizer.pad_token_type_id,
            ),
        }

        self.data = self.load(
            path, tokenizer=tokenizer, pre_process=pre_process, shuffle=shuffle
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
        # the actual data will be here
        data = []

        if isinstance(paths, Sequence):
            paths = [self.project_folder / path for path in paths]
        else:
            paths = [self.project_folder / paths]

        tmp_data = []
        # read the data and put it in a placeholder list
        for path in paths:
            if not path.exists():
                raise ValueError(f"{path} does not exist")

            # logger.log(
            #     f"Loading [bold cyan]{self.name}[/bold cyan] data from [bold]{path}[/bold]"
            # )
            # json_data = json.load(open(path, "r"))
            # tmp_data += json_data
            # The data is a list of dictionaries, each dictionary is a sample
            # Each sample has the following keys:
            #   - "question": the question
            #   - "answers": a list of answers
            #   - "positive_ctxs": a list of positive contexts
            #   - "negative_ctxs": a list of negative contexts
            #   - "hard_negative_ctxs": a list of hard negative contexts

        # use the huggingface dataset library to load the data, by default it will load the
        # data in a dict with the key being "train"
        data = load_dataset("json", data_files=[str(p) for p in paths])["train"]

        if pre_process:
            if not tokenizer:
                raise ValueError("Tokenizer is required for pre-processing")
            # Pre-process the data
            if shuffle:
                # shuffle the data
                # random.shuffle(tmp_data)
                data = data.shuffle(seed=42)

            # # measure how long the preprocessing takes
            start = time.time()
            # fn = partial(
            #     DPRDataset.process_sample,
            #     tokenizer=tokenizer,
            #     max_positives=self.max_positives,
            #     max_negatives=self.max_negatives,
            #     max_hard_negatives=self.max_hard_negatives,
            # )
            # data = fn(tmp_data)
            data = data.map(
                partial(
                    DPRDataset.process_sample,
                    tokenizer=tokenizer,
                    max_positives=self.max_positives,
                    max_negatives=self.max_negatives,
                    max_hard_negatives=self.max_hard_negatives,
                ),
                remove_columns=[
                    "answers",
                    "positive_ctxs",
                    "negative_ctxs",
                    "hard_negative_ctxs",
                ],
                keep_in_memory=True,
                load_from_cache_file=True,
                num_proc=psutil.cpu_count(logical=False),
                # num_proc=1,
            )
            # num_cores = psutil.cpu_count(logical=False)
            # chunks = [tmp_data[i::num_cores] for i in range(num_cores)]
            # with multiprocessing.Pool(processes=num_cores) as pool:
            #     results = pool.map(fn, chunks)
            # data = [item for sublist in results for item in sublist]
            end = time.time()
            logger.log(
                f"Pre-processing [bold cyan]{self.name}[/bold cyan] "
                f"data took [bold]{end - start:.2f}[/bold] seconds"
            )
        return data

    @staticmethod
    def process_sample(
        samples: List[Dict],
        tokenizer: tr.PreTrainedTokenizer,
        max_positives: int,
        max_negatives: int,
        max_hard_negatives: int,
        max_question_length: int = 256,
        max_context_length: int = 128,
    ) -> List[Dict]:
        data = []

        # questions = tokenizer(
        #     samples["question"], max_length=max_question_length, truncation=True
        # )
        # positive_ctxs = [
        #     [
        #         tokenizer(p["text"], max_length=max_context_length, truncation=True)
        #         for p in positives
        #     ]
        #     for positives in samples["positive_ctxs"]
        # ]
        # if max_positives != -1:
        #     positive_ctxs = [positives[:max_positives] for positives in positive_ctxs]
        # negative_ctxs = [
        #     [
        #         tokenizer(n["text"], max_length=max_context_length, truncation=True)
        #         for n in negatives
        #     ]
        #     for negatives in samples["negative_ctxs"]
        # ]
        # if max_negatives != -1:
        #     negative_ctxs = [negatives[:max_negatives] for negatives in negative_ctxs]
        # hard_negative_ctxs = [
        #     [
        #         tokenizer(n["text"], max_length=max_context_length, truncation=True)
        #         for n in hard_negatives
        #     ]
        #     for hard_negatives in samples["hard_negative_ctxs"]
        # ]
        # if max_hard_negatives != -1:
        #     hard_negative_ctxs = [
        #         hard_negatives[:max_hard_negatives]
        #         for hard_negatives in hard_negative_ctxs
        #     ]
        # contexts = []
        # for i in range(len(samples)):
        #     contexts_to_add = positive_ctxs[i]
        #     if i < len(negative_ctxs):
        #         contexts_to_add += negative_ctxs[i]
        #     if i < len(hard_negative_ctxs):
        #         contexts_to_add += hard_negative_ctxs[i]
        #     contexts.append(contexts_to_add)

        # data = {
        #     "question": questions,
        #     "context": contexts,
        #     "positive_ctxs": positive_ctxs,
        #     "negative_ctxs": negative_ctxs,
        #     "hard_negative_ctxs": hard_negative_ctxs,
        # }

        # for sample in track(samples, description="Pre-processing samples"):
        question = tokenizer(
            samples["question"], max_length=max_question_length, truncation=True
        )
        positive_ctxs = [
            tokenizer(p["text"], max_length=max_context_length, truncation=True)
            for p in samples["positive_ctxs"]
        ]
        if max_positives != -1:
            positive_ctxs = positive_ctxs[:max_positives]
        negative_ctxs = [
            tokenizer(n["text"], max_length=max_context_length, truncation=True)
            for n in samples["negative_ctxs"]
        ]
        if max_negatives != -1:
            negative_ctxs = negative_ctxs[:max_negatives]
        hard_negative_ctxs = [
            tokenizer(h["text"], max_length=max_context_length, truncation=True)
            for h in samples["hard_negative_ctxs"]
        ]
        if max_hard_negatives != -1:
            hard_negative_ctxs = hard_negative_ctxs[:max_hard_negatives]
        context = positive_ctxs + negative_ctxs + hard_negative_ctxs
        # data.append(
        #     {
        #         "question": question,
        #         "context": context,
        #         "positives": set([p["text"] for p in samples["positive_ctxs"]]),
        #         "positive_index_end": len(positive_ctxs),
        #     }
        # )
        # return data
        return {
                "question": question,
                "context": context,
                "positives": set([p["text"] for p in samples["positive_ctxs"]]),
                "positive_index_end": len(positive_ctxs),
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

    # @staticmethod
    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        questions = [sample["question"] for sample in batch]
        contexts = [sample["context"] for sample in batch]

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

        augmented_labels = None
        if self.in_batch_positives_augmentation:
            # labels is a mask of positive contexts for each question
            # has shape num_questions x num_contexts
            augmented_labels = torch.zeros(
                questions["input_ids"].shape[0], contexts["input_ids"].shape[0]
            )
            flat_positives = [s for sample in batch for s in sample["positives"]]
            positives = [sample["positives"] for sample in batch]
            # labels includes as positive also the labels that appear in the
            # other samples but are positive for the considered one (avoid false
            # negative context)
            for p_idx, p in enumerate(flat_positives):
                for i, positive in enumerate(positives):
                    for positive_ctx in positive:
                        if positive_ctx in p:
                            augmented_labels[i, p_idx] = 1

        model_inputs = {
            "questions": ModelInputs(questions),
            "contexts": ModelInputs(contexts),
            "labels": augmented_labels if augmented_labels is not None else labels,
            "labels_for_metrics": labels,
        }
        return ModelInputs(model_inputs)
