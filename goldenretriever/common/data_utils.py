import base64
import json
import os
import platform
import random
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Union,
)

import psutil
import transformers as tr
from streaming import MDSWriter
from streaming.base.format import get_index_basename
from tqdm import tqdm

from goldenretriever.common.hf_utils import build_hf_dataset
from goldenretriever.common.log import get_logger
from goldenretriever.common.utils import (
    GOLDENRETRIEVER_CACHE_DIR,
    file_exists,
    url_to_filename,
)

logger = get_logger(__name__)


lock = threading.Lock()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(
                        *args, **kwargs
                    )
        return cls._instances[cls]


class HardNegativesManagerThread(metaclass=Singleton):

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
        self.max_length = max_length
        self.batch_size = batch_size

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

        # invert the db to have a passage -> sample_idx mapping
        self._passage_db = defaultdict(set)
        for sample_idx, passages in self._db.items():
            for passage in passages:
                self._passage_db[passage].add(sample_idx)

        self._passage_hard_negatives = {}

    def __len__(self) -> int:
        return len(self._db)

    def __getitem__(self, idx: int) -> Dict:
        return self._db[idx]

    def __iter__(self):
        for sample in self._db:
            yield sample

    def __contains__(self, idx: int) -> bool:
        return idx in self._db

    def tokenize(self):
        # create a dictionary of passage -> hard_negative mapping
        batch_size = min(self.batch_size, len(self._passage_db))
        unique_passages = list(self._passage_db.keys())
        for i in tqdm(
            range(0, len(unique_passages), batch_size),
            desc="Tokenizing Hard Negatives",
        ):
            batch = unique_passages[i : i + batch_size]
            tokenized_passages = self.tokenizer(
                batch,
                max_length=self.max_length,
                truncation=True,
            )
            for i, passage in enumerate(batch):
                self._passage_hard_negatives[passage] = {
                    k: tokenized_passages[k][i] for k in tokenized_passages.keys()
                }

    def _add(self, idx: int, passages: List[str]) -> int:
        """
        Add a sample to the database.

        Args:
            idx (`int`): sample index
            passages (`List[str]`): list of passages
        """
        if idx in self._db:
            return idx
        self._db[idx] = passages
        for passage in passages:
            self._passage_db[passage].add(idx)
        return idx

    def add(self, idx: int | List[int], passages: List[List[str]]) -> List[int]:
        """
        Add multiple samples to the database.

        Args:
            idx (`int` | `List[int]`): sample index
            passages (`List[List[str]]`): list of passages
        """
        if isinstance(idx, int):
            idx = [idx]
            passages = [passages]
        if len(idx) != len(passages):
            raise ValueError("Length of idx and passages should be the same.")
        return [self._add(i, p) for i, p in zip(idx, passages)]

    def get(self, idx: int) -> List[str]:
        """Get the hard negatives for a given sample index."""
        if idx not in self._db:
            raise ValueError(f"Sample index {idx} not in the database.")

        passages = self._db[idx]

        output = []
        for passage in passages:
            if passage not in self._passage_hard_negatives:
                self._passage_hard_negatives[passage] = self._tokenize(passage)
            output.append(self._passage_hard_negatives[passage])

        return output

    def reset(self):
        self._db = {}
        self._passage_db = defaultdict(set)
        self._passage_hard_negatives = {}

    def _tokenize(self, passage: str) -> Dict:
        return self.tokenizer(passage, max_length=self.max_length, truncation=True)


def batch_generator(samples: Iterable[Any], batch_size: int) -> Iterable[Any]:
    """
    Generate batches from samples.

    Args:
        samples (`Iterable[Any]`): Iterable of samples.
        batch_size (`int`): Batch size.

    Returns:
        `Iterable[Any]`: Iterable of batches.
    """
    batch = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []

    # leftover batch
    if len(batch) > 0:
        yield batch


def preprocess_to_mds(
    source: str | os.PathLike,
    tokenizer_fn: callable = None,
    cache_dir: str | os.PathLike | None = None,
    use_cache: bool = True,
    num_workers: int | None = None,
) -> str | os.PathLike:
    """Preprocess the dataset to a markdown file.

    Args:
        source (str | os.PathLike): The source file or directory to preprocess.
    """

    # if os.path.exists(source):

    source = Path(source)

    if source.is_dir():
        basename = get_index_basename()
        # filename = os.path.join(self.local, self.split, basename)  # pyright: ignore
        hashed_filename = source / basename
        if hashed_filename.exists():
            logger.info(f"Found existing index file {hashed_filename}")
            return source

    # No index.json file found, so we need to create it
    if cache_dir is None:
        cache_dir = GOLDENRETRIEVER_CACHE_DIR
    cache_dir = Path(cache_dir)
    # check if cache dir exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    # get filename from the url
    hashed_filename = url_to_filename(str(source), None)
    if not use_cache:
        hashed_filename = f"{hashed_filename}.{random.randint(0, 1000000)}"
    # get cache path to put the file
    cache_path = cache_dir / hashed_filename

    # the file is already here, return it
    if use_cache and file_exists(cache_path):  # and not force_download:
        logger.info(
            f"{source} found in cache."  # , set `force_download=True` to force the download"
        )
        return str(cache_path)

    # else:
    #     source = source

    logger.info("Converting dataset to MDS format")
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if "linux" in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    dataset = build_hf_dataset(
        dataset_name=str(source),
        data_subset=None,
        split="train",
        shuffle=False,
        num_workers=num_workers,
    )
    if tokenizer_fn is not None:
        dataset = dataset.map(tokenizer_fn, desc="Tokenizing data")

    columns = {col: "pkl" for col in dataset.column_names}
    with MDSWriter(columns=columns, out=str(cache_path)) as out:
        for sample in tqdm(dataset, desc=f"Converting {source} to MDS"):
            out.write(sample)

    return str(cache_path)
