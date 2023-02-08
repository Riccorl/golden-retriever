import json
import os
from pathlib import Path
import random
from typing import Any, Tuple, Sequence
from typing import Dict, Iterator, List, Union

import torch
import transformers as tr
from rich.progress import track
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

from utils.logging import get_console_logger

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
        split_contexts: bool = False,
        max_negatives: bool = False,
        max_hard_negatives: bool = False,
        shuffle_negative_contexts: bool = False,
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)
        self.split_contexts = split_contexts
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.shuffle_negative_contexts = shuffle_negative_contexts
        self.data = self.load(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.data[index]

    def load(
        self,
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        *args,
        **kwargs,
    ) -> Any:
        # the actual data will be here
        data = []

        if isinstance(paths, Sequence):
            paths = [self.project_folder / path for path in paths]
        else:
            paths = [self.project_folder / paths]

        # read the data and put it in a placeholder list
        for path in paths:
            if not path.exists():
                raise ValueError(f"{path} does not exist")

            json_data = json.load(open(path, "r"))
            # The data is a list of dictionaries, each dictionary is a sample
            # Each sample has the following keys:
            #   - "question": the question
            #   - "answers": a list of answers
            #   - "positive_ctxs": a list of positive contexts
            #   - "negative_ctxs": a list of negative contexts
            #   - "hard_negative_ctxs": a list of hard negative contexts

            # Expand the data to have one sample per context
            logger.log(f"Loading [bold cyan]{self.name}[/bold cyan] data from [bold]{path}[/bold]")
            for sample in track(json_data):
                if self.split_contexts:
                    positive_ctxs = sample["positive_ctxs"]
                    negative_ctxs = sample["negative_ctxs"]
                    hard_negative_ctxs = sample["hard_negative_ctxs"]
                    for positive_ctx in positive_ctxs:
                        data.append(
                            {
                                "question": sample["question"],
                                "contex": positive_ctx["text"],
                                "is_positive": True,
                            }
                        )
                    if self.shuffle_negative_contexts:
                        random.shuffle(negative_ctxs)
                        random.shuffle(hard_negative_ctxs)
                    for negative_ctx in negative_ctxs[: self.max_negatives]:
                        data.append(
                            {
                                "question": sample["question"],
                                "contex": negative_ctx["text"],
                                "is_positive": False,
                            }
                        )
                    for hard_negative_ctx in hard_negative_ctxs[
                        : self.max_hard_negatives
                    ]:
                        data.append(
                            {
                                "question": sample["question"],
                                "contex": hard_negative_ctx["text"],
                                "is_positive": False,
                            }
                        )
                else:
                    data.append(
                        {
                            "question": sample["question"],
                            "positive_ctxs": [
                                ctx["text"] for ctx in sample["positive_ctxs"]
                            ],
                            "negative_ctxs": [
                                ctx["text"] for ctx in sample["negative_ctxs"]
                            ],
                            "hard_negative_ctxs": [
                                ctx["text"] for ctx in sample["hard_negative_ctxs"]
                            ],
                        }
                    )

        return data

    @staticmethod
    def collate_fn(
        batch: Any, tokenizer: tr.PreTrainedTokenizer, *args, **kwargs
    ) -> Any:
        questions = [sample["question"] for sample in batch]
        contexts = [sample["contex"] for sample in batch]

        # tokenize the questions, positive and negative contexts
        questions = tokenizer(
            questions, padding=True, truncation=True, return_tensors="pt"
        )
        # now tokenize the contexts
        contexts = tokenizer(
            contexts, padding=True, truncation=True, return_tensors="pt"
        )
        # build the labels for the positive contexts
        labels = torch.tensor(
            [i for i, sample in enumerate(batch) if sample["is_positive"]],
            dtype=torch.long,
        )
        return {"questions": questions, "contexts": contexts, "labels": labels}
