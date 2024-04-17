from functools import partial
import os
from typing import Any, List, Optional, Sequence, Union

import hydra
import lightning as pl
from litdata import StreamingDataLoader
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from goldenretriever.common.log import get_logger
from goldenretriever.data.datasets import GoldenRetrieverDataset

# from goldenretriever.data.streaming_dataset import GoldenRetrieverCollator, StreamingGoldenRetrieverDataset
from goldenretriever.data.lit_dataset import GoldenStreamingDataset
from goldenretriever.data.streaming_dataset import (
    GoldenRetrieverCollator,
    GoldenRetrieverStreamingDataset,
)
from goldenretriever.data.utils import (
    GoldenDistributedSampler,
    HardNegativesManagerThread,
)
from goldenretriever.data.utils import HardNegativesManagerThread

import transformers as tr

logger = get_logger(__name__)


class GoldenRetrieverPLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Optional[GoldenRetrieverDataset] = None,
        train_dataset_kwargs: dict = None,
        val_datasets: Optional[Sequence[GoldenRetrieverDataset]] = None,
        val_datasets_kwargs: List[dict] = None,
        test_datasets: Optional[Sequence[GoldenRetrieverDataset]] = None,
        test_datasets_kwargs: List[dict] = None,
        num_workers: Optional[Union[DictConfig, int]] = None,
        datasets: Optional[DictConfig] = None,
        tokenizer=None,
        question_tokenizer=None,
        passage_tokenizer=None,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.datasets = datasets
        if num_workers is None:
            num_workers = 0
        if isinstance(num_workers, int):
            num_workers = DictConfig(
                {"train": num_workers, "val": num_workers, "test": num_workers}
            )
        self.num_workers = num_workers
        # kwargs
        self.train_dataset_kwargs = train_dataset_kwargs or {}
        if val_datasets is not None:
            self.val_datasets_kwargs = val_datasets_kwargs or [{} for _ in val_datasets]
        self.val_datasets_kwargs = val_datasets_kwargs
        if test_datasets is not None:
            self.test_datasets_kwargs = test_datasets_kwargs or [
                {} for _ in test_datasets
            ]
        # data
        self.train_dataset: Optional[GoldenRetrieverStreamingDataset] = train_dataset
        self.val_datasets: Optional[Sequence[GoldenRetrieverStreamingDataset]] = (
            val_datasets
        )
        self.test_datasets: Optional[Sequence[GoldenRetrieverStreamingDataset]] = (
            test_datasets
        )
        self.tokenizer = tokenizer
        self.question_tokenizer = question_tokenizer or tokenizer
        self.passage_tokenizer = passage_tokenizer or tokenizer

        # other stuff
        self.seed = seed

    def prepare_data(self, *args, **kwargs):
        """
        Method for preparing the data before the training. This method is called only once.
        It is used to download the data, tokenize the data, etc.
        """
        # preprocess dataset
        if self.train_dataset is not None:
            data_path = None
            if isinstance(self.train_dataset, (str, os.PathLike)):
                data_path = self.train_dataset
            elif isinstance(self.train_dataset, DictConfig()):
                # TODO
                data_path = self.train_dataset["local"]
            else:
                logger.debug("No data path found, skipping preprocessing")
            GoldenRetrieverStreamingDataset.preprocess_to_mds(
                data_path,
                partial(
                    GoldenRetrieverStreamingDataset.tokenize,
                    **{
                        "question_tokenizer": self.question_tokenizer,
                        "passage_tokenizer": self.passage_tokenizer,
                        **self.train_dataset_kwargs,
                    },
                ),
            )

        if self.val_datasets is not None:
            for i, dataset in enumerate(self.val_datasets):
                data_path = None
                if isinstance(dataset, (str, os.PathLike)):
                    data_path = dataset
                elif isinstance(dataset, DictConfig()):
                    # TODO
                    data_path = dataset["local"]
                else:
                    logger.debug("No data path found, skipping preprocessing")
                GoldenRetrieverStreamingDataset.preprocess_to_mds(
                    data_path,
                    partial(
                        GoldenRetrieverStreamingDataset.tokenize,
                        **{
                            "question_tokenizer": self.question_tokenizer,
                            "passage_tokenizer": self.passage_tokenizer,
                            **self.val_datasets_kwargs[i],
                        },
                    ),
                )

        if self.test_datasets is not None:
            for dataset in self.test_datasets:
                data_path = None
                if isinstance(dataset, (str, os.PathLike)):
                    data_path = dataset
                elif isinstance(dataset, DictConfig()):
                    # TODO
                    data_path = dataset["local"]
                else:
                    logger.debug("No data path found, skipping preprocessing")
                GoldenRetrieverStreamingDataset.preprocess_to_mds(
                    data_path,
                    partial(
                        GoldenRetrieverStreamingDataset.tokenize,
                        **{
                            "question_tokenizer": self.question_tokenizer,
                            "passage_tokenizer": self.passage_tokenizer,
                            **self.val_datasets_kwargs[i],
                        },
                    ),
                )

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            # usually there is only one dataset for train
            # if you need more train loader, you can follow
            # the same logic as val and test datasets
            # if self.train_dataset is None:
            #     self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            #     self.val_datasets = [
            #         hydra.utils.instantiate(dataset_cfg)
            #         for dataset_cfg in self.datasets.val
            #     ]
            # self.train_dataset = GoldenRetrieverStreamingDataset(
            #     name="aida_train",
            #     question_tokenizer=self.tokenizer,
            #     local="/home/ric/Projects/golden-retriever/data/dpr-like/el/mosaic/train",
            #     batch_size=64,
            #     predownload=64 * 64,
            #     shuffle_seed=42,
            # )
            # self.train_dataset = GoldenStreamingDataset(
            #     name="streaming_train",
            #     question_tokenizer=self.tokenizer,
            #     input_dir="/home/ric/Projects/golden-retriever/data/dpr-like/el/litdata/train",
            # )
            if isinstance(self.train_dataset, (str, os.PathLike)):
                self.train_dataset, self.train_dataset_kwargs = self.dataset_builder(
                    dataset=self.train_dataset,
                    name="train_dataset",
                    question_tokenizer=self.question_tokenizer,
                    passage_tokenizer=self.passage_tokenizer,
                    shuffle_seed=self.seed,
                    dataset_kwargs=self.train_dataset_kwargs,
                    shuffle=True,
                )
            elif isinstance(self.train_dataset, DictConfig):
                self.train_dataset = hydra.utils.instantiate(self.train_dataset)
            else:
                self.train_dataset = self.train_dataset

            _val_dataset = []
            # keep track also of the kwargs
            _val_dataset_kwargs = []
            for i, dataset in enumerate(self.val_datasets):
                if isinstance(dataset, (str, os.PathLike)):
                    val_dataset, ds_kwargs = self.dataset_builder(
                        dataset=dataset,
                        name=f"val_dataset_{i}",
                        question_tokenizer=self.question_tokenizer,
                        passage_tokenizer=self.passage_tokenizer,
                        shuffle=False,
                        shuffle_seed=self.seed,
                        dataset_kwargs=self.val_datasets_kwargs[i],
                    )
                elif isinstance(dataset, DictConfig):
                    val_dataset = hydra.utils.instantiate(dataset)
                else:
                    val_dataset = dataset

                _val_dataset.append(val_dataset)
                # keep track of the kwargs
                _val_dataset_kwargs.append(ds_kwargs)

            # update val_dataset with the new datasets
            self.val_datasets = _val_dataset
            # update val_dataset_kwargs with the new kwargs
            self.val_datasets_kwargs = _val_dataset_kwargs

            # self.val_dataset = GoldenRetrieverStreamingDataset(
            #     name="aida_val",
            #     question_tokenizer=self.tokenizer,
            #     local="/home/ric/Projects/golden-retriever/data/dpr-like/el/mosaic/val",
            #     batch_size=64,
            #     predownload=64 * 64,
            #     shuffle_seed=42,
            # )
            # self.val_dataset = GoldenStreamingDataset(
            #     name="streaming_val",
            #     question_tokenizer=self.tokenizer,
            #     input_dir="/home/ric/Projects/golden-retriever/data/dpr-like/el/litdata/val",
            # )
            # self.val_datasets = [self.val_dataset]
        if stage == "test":
            if self.test_datasets is None:
                self.test_datasets = [
                    hydra.utils.instantiate(dataset_cfg)
                    for dataset_cfg in self.datasets.test
                ]

            _test_dataset = []
            # keep track also of the kwargs
            _test_dataset_kwargs = []
            for i, dataset in enumerate(self.test_datasets):
                if isinstance(dataset, (str, os.PathLike)):
                    test_dataset, ds_kwargs = self.dataset_builder(
                        dataset=dataset,
                        name=f"test_dataset_{i}",
                        question_tokenizer=self.question_tokenizer,
                        passage_tokenizer=self.passage_tokenizer,
                        shuffle=False,
                        shuffle_seed=self.seed,
                        dataset_kwargs=self.test_datasets_kwargs[i],
                    )
                elif isinstance(dataset, DictConfig):
                    test_dataset = hydra.utils.instantiate(dataset)
                else:
                    test_dataset = dataset

                _test_dataset.append(test_dataset)
                # keep track of the kwargs
                _test_dataset_kwargs.append(ds_kwargs)

            # update val_dataset with the new datasets
            self.test_datasets = _test_dataset
            # update val_dataset_kwargs with the new kwargs
            self.test_datasets_kwargs = _test_dataset_kwargs

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            collate_fn=GoldenRetrieverCollator(
                tokenizer=self.train_dataset.question_tokenizer,
                max_passage_length=self.train_dataset.max_passage_length,
            ),
            shuffle=True,
            batch_size=self.train_dataset_kwargs.get("batch_size"),
            num_workers=self.num_workers.train,
            pin_memory=True,
            prefetch_factor=(
                max(1, 8 * self.train_dataset.batch_size // self.num_workers.train)
                if self.num_workers.train > 0
                else None
            ),
            persistent_workers=True if self.num_workers.train > 0 else False,
            timeout=0,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataloaders = []
        for i, dataset in enumerate(self.val_datasets):
            dataloaders.append(
                DataLoader(
                    dataset,
                    collate_fn=GoldenRetrieverCollator(
                        tokenizer=dataset.question_tokenizer,
                        max_passage_length=dataset.max_passage_length,
                    ),
                    shuffle=False,
                    batch_size=self.val_datasets_kwargs[i].get("batch_size"),
                    num_workers=self.num_workers.val,
                    pin_memory=True,
                    prefetch_factor=(
                        max(
                            1, 8 * self.train_dataset.batch_size // self.num_workers.val
                        )
                        if self.num_workers.val > 0
                        else None
                    ),
                    persistent_workers=True if self.num_workers.val > 0 else False,
                    timeout=0,
                )
            )
        return dataloaders

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataloaders = []
        for i, dataset in enumerate(self.test_datasets):
            dataloaders.append(
                DataLoader(
                    dataset,
                    collate_fn=GoldenRetrieverCollator(
                        tokenizer=dataset.question_tokenizer,
                        max_passage_length=dataset.max_passage_length,
                    ),
                    shuffle=False,
                    batch_size=self.test_datasets_kwargs[i].get("batch_size"),
                    num_workers=self.num_workers.val,
                    pin_memory=True,
                    prefetch_factor=(
                        max(
                            1,
                            8 * self.train_dataset.batch_size // self.num_workers.test,
                        )
                        if self.num_workers.test > 0
                        else None
                    ),
                    persistent_workers=True if self.num_workers.test > 0 else False,
                    timeout=0,
                )
            )
        return dataloaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, "
        )

    @staticmethod
    def dataset_builder(
        dataset: str | GoldenRetrieverStreamingDataset = None,
        name: str = None,
        batch_size: int = None,
        question_tokenizer: tr.PreTrainedTokenizerBase = None,
        passage_tokenizer: tr.PreTrainedTokenizerBase = None,
        shuffle: bool = None,
        shuffle_seed: int = None,
        dataset_kwargs: dict = None,
    ):
        dataset = dataset or dataset_kwargs.get("local", None)
        if dataset is None:
            raise ValueError("The dataset is required.")
        if isinstance(dataset, str):
            # check if all the necessary parameters are provided
            if name is None and "name" not in dataset_kwargs:
                raise ValueError("The dataset name is required.")
            if batch_size is None and "batch_size" not in dataset_kwargs:
                raise ValueError("The batch size is required.")
            if (
                question_tokenizer is None
                and "question_tokenizer" not in dataset_kwargs
            ):
                raise ValueError("The question_tokenizer is required.")
            if passage_tokenizer is None and "passage_tokenizer" not in dataset_kwargs:
                raise ValueError("The passage_tokenizer is required.")
            if shuffle is None and "shuffle" not in dataset_kwargs:
                raise ValueError("The shuffle parameter is required.")
            if shuffle_seed is None and "shuffle_seed" not in dataset_kwargs:
                raise ValueError("The shuffle_seed parameter is required.")

            if "name" not in dataset_kwargs:
                dataset_kwargs["name"] = name
            if "local" not in dataset_kwargs:
                dataset_kwargs["local"] = dataset
            if "question_tokenizer" not in dataset_kwargs:
                dataset_kwargs["question_tokenizer"] = question_tokenizer
            if "passage_tokenizer" not in dataset_kwargs:
                dataset_kwargs["passage_tokenizer"] = passage_tokenizer
            if "batch_size" not in dataset_kwargs:
                dataset_kwargs["batch_size"] = batch_size
            if "shuffle" not in dataset_kwargs:
                dataset_kwargs["shuffle"] = shuffle
            if "shuffle_seed" not in dataset_kwargs:
                dataset_kwargs["shuffle_seed"] = shuffle_seed
            dataset = GoldenRetrieverStreamingDataset(**dataset_kwargs)

        return dataset, dataset_kwargs
