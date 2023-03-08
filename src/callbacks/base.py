import enum
from typing import Any, Dict, List, Optional, Set, Union

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from utils.logging import get_console_logger

# from faiss.indexer import FaissIndexer

logger = get_console_logger()


class Stage(enum.Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class NLPTemplateCallback:
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        predictions: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError


class PredictionCallback(pl.Callback):
    def __init__(
        self,
        stages: Set[Union[str, Stage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
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
            callback(trainer, pl_module, Stage.VALIDATION, predictions)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        predictions = self(trainer, pl_module, Stage.TEST)
        for callback in self.other_callbacks:
            callback(trainer, pl_module, Stage.TEST, predictions)
