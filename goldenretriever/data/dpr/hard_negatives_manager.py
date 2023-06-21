import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Union
from datasets import Dataset, load_dataset
import psutil
import transformers as tr


class HardNegativeManager:
    def __init__(
        self,
        data: Union[Dataset, List[Dict], os.PathLike],
        tokenizer: tr.PreTrainedTokenizer,
        max_length: int = 64,
    ) -> None:
        self._db: Dataset = None
        self.tokenizer = tokenizer

        if isinstance(data, Dataset):
            self._db = data
        elif isinstance(data, list):
            # convert the dictionary to a dataset for easy multiprocessing
            # dump in a temporary file and load it again into a dataset
            # this is necessary because the dataset cannot load in-memory data
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = Path(tmp_dir) / "hard_negatives.jsonl"
                with open(tmp_file, "w") as f:
                    for sample in data:
                        f.write(json.dumps(sample) + "\n")
                # clean hard_negatives_dict
                data = None
                # load the dataset from the temporary file
                self._db = load_dataset(
                    "json", data_files=[str(tmp_file)], split="train"
                )
        elif isinstance(data, os.PathLike):
            self._db = load_dataset("json", data_files=[str(data)], split="train")
        else:
            raise ValueError(
                f"Data type {type(data)} not supported, only Dataset, List[Dict] and os.PathLike are supported."
            )

        # map the hard negatives
        self._db = self._db.map(
            self._map,
            fn_kwargs=dict(
                tokenizer=tokenizer,
                max_length=max_length,
            ),
            num_proc=psutil.cpu_count(),
            desc="Tokenizing hard negatives",
        )

        self._db_ids = set(self._db["sample_idx"])

        # for easy indexing
        self._db = self._db.to_dict()

    def __len__(self) -> int:
        return len(self._db)

    def __getitem__(self, idx: int) -> Dict:
        return self._db[idx]

    def __iter__(self):
        for sample in self._db:
            yield sample

    def __contains__(self, idx: int) -> bool:
        return idx in self._db_ids

    def get_hard_negatives(self, idx: int) -> List[Dict]:
        if idx not in self:
            raise ValueError(f"Index {idx} not found in the db.")
        
        return self._db["hard_negatives"][idx]

    @staticmethod
    def _map(sample, tokenizer: tr.PreTrainedTokenizer, max_length: int):
        contexts = sample["contexts"]
        context_ids = tokenizer(contexts, max_length=max_length, truncation=True)
        sample["hard_negatives"] = [
            {k: v[index] for k, v in context_ids.items()}
            for index in range(len(contexts))
        ]
        return sample
