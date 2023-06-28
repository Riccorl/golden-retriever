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

from goldenretriever.common.log import get_logger
from goldenretriever.data.labels import Labels

logger = get_logger()


class PLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        batch_sizes: Optional[DictConfig] = None,
        num_workers: Optional[DictConfig] = None,
        tokenizer: Union[str, tr.PreTrainedTokenizer] = None,
        labels: Labels = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_sizes = batch_sizes
        # data
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None
        # label file
        self.labels: Labels = labels
        # tokenizer
        self.tokenizer: tr.PreTrainedTokenizer = None
        if tokenizer is not None:
            self.tokenizer = (
                tokenizer
                if isinstance(tokenizer, tr.PreTrainedTokenizer)
                else tr.AutoTokenizer.from_pretrained(tokenizer)
            )

    def build_labels(self) -> Labels:
        """
        Builds the labels for the model

        Returns:
            `Labels`: A dictionary of labels
        """
        raise NotImplementedError

    def save_labels(self, path: Union[str, os.PathLike]) -> None:
        """
        Saves the labels to a file

        Args:
            path (str): The path to save the labels to
        """
        if self.labels is None:
            logger.info("No labels to save")
            return
        self.labels.save(path)

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
                self.train_dataset = hydra.utils.instantiate(
                    self.datasets.train, tokenizer=self.tokenizer
                )
                self.val_datasets = [
                    hydra.utils.instantiate(dataset_cfg, tokenizer=self.tokenizer)
                    for dataset_cfg in self.datasets.val
                ]
        if stage == "test":
            if self.test_datasets is None:
                self.test_datasets = [
                    hydra.utils.instantiate(dataset_cfg, tokenizer=self.tokenizer)
                    for dataset_cfg in self.datasets.test
                ]

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        # torch_dataset = self.train_dataset.to_torch_dataset()
        return DataLoader(
            # torch_dataset,
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
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_sizes=})"
        )
