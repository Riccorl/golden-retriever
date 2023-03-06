import enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from utils.logging import get_console_logger
from utils.model_inputs import ModelInputs

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
        other_callbacks: Optional[List[NLPTemplateCallback]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.other_callbacks = other_callbacks or []

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
