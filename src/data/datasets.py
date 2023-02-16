import json
import os
import random
from functools import partial
from pathlib import Path
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
        shuffle: bool = False,
        pre_process: bool = True,
        split_contexts: bool = False,
        max_negatives: bool = False,
        max_hard_negatives: bool = False,
        shuffle_negative_contexts: bool = False,
        tokenizer: tr.PreTrainedTokenizer = None,
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)
        self.split_contexts = split_contexts
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.shuffle_negative_contexts = shuffle_negative_contexts

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

            logger.log(
                f"Loading [bold cyan]{self.name}[/bold cyan] data from [bold]{path}[/bold]"
            )
            json_data = json.load(open(path, "r"))
            tmp_data += json_data
            # The data is a list of dictionaries, each dictionary is a sample
            # Each sample has the following keys:
            #   - "question": the question
            #   - "answers": a list of answers
            #   - "positive_ctxs": a list of positive contexts
            #   - "negative_ctxs": a list of negative contexts
            #   - "hard_negative_ctxs": a list of hard negative contexts

        if pre_process:
            if not tokenizer:
                raise ValueError("Tokenizer is required for pre-processing")
            # Pre-process the data
            if shuffle:
                # shuffle the data
                random.shuffle(tmp_data)

            for sample in track(tmp_data):
                question = tokenizer(sample["question"])
                positive_ctxs = [tokenizer(p["text"]) for p in sample["positive_ctxs"]]
                negative_ctxs = [tokenizer(n["text"]) for n in sample["negative_ctxs"]]
                hard_negative_ctxs = [
                    tokenizer(h["text"]) for h in sample["hard_negative_ctxs"]
                ]
                context = positive_ctxs + negative_ctxs + hard_negative_ctxs
                data.append(
                    {
                        "question": question,
                        "context": context,
                        "positives": set([p["text"] for p in sample["positive_ctxs"]]),
                        "positive_indices": [
                            p_idx for p_idx in range(len(positive_ctxs))
                        ],
                    }
                )

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

    # @staticmethod
    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        questions = [sample["question"] for sample in batch]
        contexts = [sample["context"] for sample in batch]

        questions = self.convert_to_batch(questions)
        # first flat the list of list of contexts
        contexts = [c for ctxs in contexts for c in ctxs]
        # invert contexts from list of dict to dict of list
        contexts = self.convert_to_batch(contexts)

        # labels is a mask of positive contexts for each question
        # has shape num_questions x num_contexts
        labels = torch.zeros(
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
                        labels[i, p_idx] = 1

        model_inputs = {"questions": questions, "contexts": contexts, "labels": labels}

        # additional stuff
        # positive indices for computing actual recall-at-k that doesn't include
        # the other weak-positive stuff
        positive_indices = []
        for s_idx, sample in enumerate(batch):
            if s_idx == 0:
                positive_indices.append(sample["positive_indices"])
            else:
                # add the last index of the previous sample to the current one as offset
                positive_indices.append(
                    [
                        p_idx + positive_indices[-1][-1] + 1
                        for p_idx in sample["positive_indices"]
                    ]
                )
        model_inputs.update({"positive_indices": positive_indices})
        return model_inputs
