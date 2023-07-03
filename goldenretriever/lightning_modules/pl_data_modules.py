import os
from functools import partial
from typing import Any, Union, List, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
import transformers as tr
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from goldenretriever.data.datasets import GoldenRetrieverDataset
from goldenretriever.common.log import get_logger
from goldenretriever.data.labels import Labels

logger = get_logger()


class GoldenRetrieverPLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Optional[GoldenRetrieverDataset] = None,
        val_datasets: Optional[Sequence[GoldenRetrieverDataset]] = None,
        test_datasets: Optional[Sequence[GoldenRetrieverDataset]] = None,
        num_workers: Optional[Union[DictConfig, int]] = None,
        datasets: Optional[DictConfig] = None,
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
        # data
        self.train_dataset: Optional[GoldenRetrieverDataset] = train_dataset
        self.val_datasets: Optional[Sequence[GoldenRetrieverDataset]] = val_datasets
        self.test_datasets: Optional[Sequence[GoldenRetrieverDataset]] = test_datasets

    def prepare_data(self, *args, **kwargs):
        """
        Method for preparing the data before the training. This method is called only once.
        It is used to download the data, tokenize the data, etc.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # usually there is only one dataset for train
            # if you need more train loader, you can follow
            # the same logic as val and test datasets
            if self.train_dataset is None:
                self.train_dataset = hydra.utils.instantiate(self.datasets.train)
                self.val_datasets = [
                    hydra.utils.instantiate(dataset_cfg)
                    for dataset_cfg in self.datasets.val
                ]
        if stage == "test":
            if self.test_datasets is None:
                self.test_datasets = [
                    hydra.utils.instantiate(dataset_cfg)
                    for dataset_cfg in self.datasets.test
                ]

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        # torch_dataset = self.train_dataset.to_torch_dataset()
        return DataLoader(
            self.train_dataset.to_torch_dataset(),
            shuffle=False,
            batch_size=None,
            num_workers=self.num_workers.train,
            pin_memory=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [
            DataLoader(
                # dataset,
                dataset.to_torch_dataset(),
                shuffle=False,
                batch_size=None,
                num_workers=self.num_workers.val,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [
            DataLoader(
                dataset.to_torch_dataset(),
                shuffle=False,
                batch_size=None,
                num_workers=self.num_workers.test,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
            for dataset in self.test_datasets
        ]

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
