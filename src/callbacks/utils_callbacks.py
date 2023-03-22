import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple

import hydra
import pytorch_lightning as pl
import torch
import transformers as tr
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from callbacks.base import NLPTemplateCallback, PredictionCallback, Stage
from data.datasets import BaseDataset, DPRDataset
from models.model import GoldenRetriever
from utils.logging import get_console_logger
from utils.model_inputs import ModelInputs

logger = get_console_logger()

class SavePredictionCallback(NLPTemplateCallback):
    def __init__(
        self,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.verbose = verbose

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        predictions: Dict,
        callback: PredictionCallback,
        *args,
        **kwargs,
    ) -> dict:
        if self.verbose:
            logger.log(f"Computing recall@{self.k}")

        # write the predictions to a file inside the experiment folder
        if self.predictions_dir is None and trainer.logger is None:
            logger.log(
                "You need to specify an output directory (`predictions_dir`) or a logger to save the predictions."
            )
        else:
            for dataloader_idx, predictions in predictions.items():
                datasets = callback.dataset
                # save to file
                if self.save_predictions:
                    if self.predictions_dir is not None:
                        prediction_folder = Path(self.predictions_dir)
                    else:
                        prediction_folder = (
                            Path(trainer.logger.experiment.dir) / "predictions"
                        )
                        prediction_folder.mkdir(exist_ok=True)
                    predictions_path = (
                        prediction_folder
                        / f"{datasets[dataloader_idx].name}_{dataloader_idx}.json"
                    )
                    # update the dataset with the predictions
                    for sample, prediction in zip(
                        datasets[dataloader_idx], predictions
                    ):
                        sample["gold"] = prediction["gold"]
                        sample["predictions"] = prediction["predictions"]
                        sample["correct"] = prediction["correct"]
                        sample["wrong"] = prediction["wrong"]

                    logger.log(f"Saving predictions to {predictions_path}")
                    datasets[dataloader_idx].save_data(
                        predictions_path, remove_columns=self.remove_columns
                    )


class SaveRetrieverCallback(NLPTemplateCallback):
    def __init__(
        self,
        retriever_dir: Optional[Union[str, os.PathLike]] = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.retriever_dir = retriever_dir
        self.verbose = verbose

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        predictions: Dict,
        callback: PredictionCallback,
        *args,
        **kwargs,
    ) -> dict:
        if self.retriever_dir is None and trainer.logger is None:
            logger.log(
                "You need to specify an output directory (`retriever_dir`) or a logger to save the retriever."
            )
        else:
            if self.retriever_dir is not None:
                retriever_folder = Path(self.retriever_dir)
            else:
                retriever_folder = Path(trainer.logger.experiment.dir) / "retriever"
                retriever_folder.mkdir(exist_ok=True)
            if self.verbose:
                logger.log(f"Saving retriever to {retriever_folder}")
            pl_module.model.save(retriever_folder)


class FreeUpIndexerVRAMCallback(NLPTemplateCallback):
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        predictions: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Any:
        logger.log("Freeing up GPU memory")
        # remove the index from the GPU memory
        pl_module.model._context_embeddings = None
        torch.cuda.empty_cache()


class ShuffleTrainDatasetCallback(pl.Callback):
    def __init__(self, seed: int = 42, verbose: bool = True) -> None:
        super().__init__()
        self.seed = seed
        self.verbose = verbose

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if self.verbose:
            logger.log(f"Shuffling train dataset at epoch {trainer.current_epoch}")
        trainer.datamodule.train_dataset.shuffle_data(seed=self.seed + trainer.current_epoch)
