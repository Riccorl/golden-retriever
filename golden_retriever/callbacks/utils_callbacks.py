import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch

from golden_retriever.callbacks.base import NLPTemplateCallback, PredictionCallback
from golden_retriever.common.logging import get_console_logger

logger = get_console_logger()


class SavePredictionsCallback(NLPTemplateCallback):
    def __init__(
        self,
        saving_dir: Optional[Union[str, os.PathLike]] = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.saving_dir = saving_dir
        self.verbose = verbose

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Dict,
        callback: PredictionCallback,
        *args,
        **kwargs,
    ) -> dict:
        # write the predictions to a file inside the experiment folder
        if self.saving_dir is None and trainer.logger is None:
            logger.log(
                "You need to specify an output directory (`saving_dir`) or a logger to save the predictions."
            )
        else:
            datasets = callback.datasets
            for dataloader_idx, predictions in predictions.items():
                # save to file
                if self.saving_dir is not None:
                    prediction_folder = Path(self.saving_dir)
                else:
                    prediction_folder = (
                        Path(trainer.logger.experiment.dir) / "predictions"
                    )
                    prediction_folder.mkdir(exist_ok=True)
                predictions_path = (
                    prediction_folder
                    / f"{datasets[dataloader_idx].name}_{dataloader_idx}.json"
                )
                if self.verbose:
                    logger.log(f"Saving predictions to {predictions_path}")
                with open(predictions_path, "w") as f:
                    json.dump(predictions, f, indent=2)


class FreeUpIndexerVRAMCallback(NLPTemplateCallback):
    def __call__(
        self,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> Any:
        logger.log("Freeing up GPU memory")
        # remove the index from the GPU memory
        # pl_module.model._context_embeddings = None
        torch.cuda.empty_cache()


class ShuffleTrainDatasetCallback(pl.Callback):
    def __init__(self, seed: int = 42, verbose: bool = True) -> None:
        super().__init__()
        self.seed = seed
        self.verbose = verbose

    def on_validation_epoch_end(self, trainer: pl.Trainer, *args, **kwargs):
        if self.verbose:
            logger.log(
                f"Sampling negatives for train dataset at epoch {trainer.current_epoch}"
            )
        trainer.datamodule.train_dataset.sample_dataset_negatives(
            seed=self.seed + trainer.current_epoch
        )


class SaveRetrieverCallback(pl.Callback):
    def __init__(
        self,
        saving_dir: Optional[Union[str, os.PathLike]] = None,
        verbose: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.saving_dir = saving_dir
        self.verbose = verbose

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ):
        if self.saving_dir is None and trainer.logger is None:
            logger.log(
                "You need to specify an output directory (`saving_dir`) or a logger to save the retriever."
            )
        else:
            if self.saving_dir is not None:
                retriever_folder = Path(self.saving_dir)
            else:
                retriever_folder = Path(trainer.logger.experiment.dir) / "retriever"
            retriever_folder.mkdir(exist_ok=True, parents=True)
            if self.verbose:
                logger.log(f"Saving retriever to {retriever_folder}")
            pl_module.model.save_pretrained(retriever_folder)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ):
        self(trainer, pl_module)

    # def on_test_epoch_end(
    #     self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    # ) -> None:
    #     return self(trainer, pl_module)
