import enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from data.datasets import BaseDataset
from common.logging import get_console_logger

logger = get_console_logger()


class Stage(enum.Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class PredictionCallback(pl.Callback):
    def __init__(
        self,
        batch_size: int = 32,
        stages: Set[Union[str, Stage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        datasets: Optional[Union[DictConfig, BaseDataset]] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        # parameters
        self.batch_size = batch_size
        self.datasets = datasets
        self.dataloaders = dataloaders

        # callback initialization
        if stages is None:
            stages = {Stage.VALIDATION, Stage.TEST}
        self.stages = [Stage(stage) for stage in stages]
        self.other_callbacks = other_callbacks or []
        for i, callback in enumerate(self.other_callbacks):
            if isinstance(callback, DictConfig):
                self.other_callbacks[i] = hydra.utils.instantiate(
                    callback, _recursive_=False
                )

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        *args,
        **kwargs,
    ) -> Any:
        # it should return the predictions
        raise NotImplementedError

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        predictions = self(trainer, pl_module, Stage.VALIDATION)
        for callback in self.other_callbacks:
            callback(
                trainer=trainer,
                pl_module=pl_module,
                stage=Stage.VALIDATION,
                callback=self,
                predictions=predictions,
            )

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        predictions = self(trainer, pl_module, Stage.TEST)
        for callback in self.other_callbacks:
            callback(
                trainer=trainer,
                pl_module=pl_module,
                stage=Stage.TEST,
                callback=self,
                predictions=predictions,
            )

    @staticmethod
    def _get_datasets_and_dataloaders(
        dataset: Optional[Union[Dataset, DictConfig]],
        dataloader: Optional[DataLoader],
        stage: Stage,
        trainer: pl.Trainer,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        collate_fn: Optional[Callable] = None,
        collate_fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dataset], List[DataLoader]]:
        """
        Get the datasets and dataloaders from the datamodule or from the dataset provided.

        Args:
            dataset (`Optional[Union[Dataset, DictConfig]]`):
                The dataset to use. If `None`, the datamodule is used.
            batch_size (`int`):
                The batch size to use for the dataloaders.
            num_workers (`int`):
                The number of workers to use for the dataloaders.
            stage (`Stage`):
                The stage that indicates whether the dataloaders are for validation or testing.
            trainer (`pl.Trainer`):
                The trainer that contains the datamodule.
            tokenizer (`tr.PreTrainedTokenizer`):
                The tokenizer to use for the dataloaders.

        Returns:
            `Tuple[List[Dataset], List[DataLoader]]`: The datasets and dataloaders.
        """
        # if a dataset is provided, use it
        if dataset is not None:
            dataloader_kwargs = dataloader_kwargs or {}
            # get dataset
            if isinstance(dataset, DictConfig):
                dataset = hydra.utils.instantiate(dataset, _recursive_=False)
            datasets = [dataset] if isinstance(dataset, Dataset) else dataset
            if dataloader is not None:
                dataloaders = (
                    [dataloader] if isinstance(dataloader, DataLoader) else dataloader
                )
            else:
                collate_fn = collate_fn or partial(
                    datasets[0].collate_fn, **collate_fn_kwargs
                )
                dataloader_kwargs["collate_fn"] = collate_fn
                dataloaders = [DataLoader(datasets[0], **dataloader_kwargs)]
        else:
            # get the dataloaders and datasets from the datamodule
            datasets = (
                trainer.datamodule.val_datasets
                if stage == Stage.VALIDATION
                else trainer.datamodule.test_datasets
            )
            dataloaders = (
                trainer.val_dataloaders
                if stage == Stage.VALIDATION
                else trainer.test_dataloaders
            )
        return datasets, dataloaders


class NLPTemplateCallback:
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        callback: PredictionCallback,
        predictions: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError
