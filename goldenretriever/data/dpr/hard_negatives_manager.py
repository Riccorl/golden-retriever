from collections import defaultdict
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Union
from datasets import Dataset, load_dataset
import psutil
import transformers as tr

import concurrent.futures


class HardNegativeManager:
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

        # invert the db to have a context -> sample_idx mapping
        self._context_db = defaultdict(set)
        for sample_idx, contexts in self._db.items():
            for context in contexts:
                self._context_db[context].add(sample_idx)

        self._context_hard_negatives = {}
        if not lazy:
            # create a dictionary of context -> hard_negative mapping
            batch_size = min(batch_size, len(self._context_db))
            unique_contexts = list(self._context_db.keys())
            for i in range(0, len(unique_contexts), batch_size):
                batch = unique_contexts[i : i + batch_size]
                tokenized_contexts = self.tokenizer(
                    batch,
                    max_length=max_length,
                    truncation=True,
                )
                for i, context in enumerate(batch):
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
