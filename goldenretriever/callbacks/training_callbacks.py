import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence, Union

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.states import RunningStage
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from goldenretriever.callbacks.prediction_callbacks import (
    GoldenRetrieverPredictionCallback,
)
from goldenretriever.common.log import get_logger
from goldenretriever.data.base.datasets import BaseDataset

# from goldenretriever.data.streaming_dataset import GoldenRetrieverCollator
from goldenretriever.data.streaming_dataset import (
    GoldenRetrieverCollator,
    GoldenRetrieverStreamingDataset,
    GoldenStreamingDataLoader,
)
from goldenretriever.data.utils import HardNegativesManager, HardNegativesManagerThread
from goldenretriever.pytorch_modules.model import GoldenRetriever
import goldenretriever.common.dist_utils as dist

from streaming.base.util import clean_stale_shared_memory

logger = get_logger(__name__, level=logging.INFO)


class NegativeAugmentationCallback(GoldenRetrieverPredictionCallback):
    """
    Callback that computes the predictions of a retriever model on a dataset and computes the
    negative examples for the training set.

    Args:
        k (:obj:`int`, `optional`, defaults to 100):
            The number of top-k retrieved passages to
            consider for the evaluation.
        batch_size (:obj:`int`, `optional`, defaults to 32):
            The batch size to use for the evaluation.
        num_workers (:obj:`int`, `optional`, defaults to 0):
            The number of workers to use for the evaluation.
        force_reindex (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to force the reindexing of the dataset.
        retriever_dir (:obj:`Path`, `optional`):
            The path to the retriever directory. If not specified, the retriever will be
            initialized from scratch.
        stages (:obj:`Set[str]`, `optional`):
            The stages to run the callback on. If not specified, the callback will be run on
            train, validation and test.
        other_callbacks (:obj:`List[DictConfig]`, `optional`):
            A list of other callbacks to run on the same stages.
        dataset (:obj:`Union[DictConfig, BaseDataset]`, `optional`):
            The dataset to use for the evaluation. If not specified, the dataset will be
            initialized from scratch.
        metrics_to_monitor (:obj:`List[str]`, `optional`):
            The metrics to monitor for the evaluation.
        threshold (:obj:`float`, `optional`, defaults to 0.8):
            The threshold to consider. If the recall score of the retriever is above the
            threshold, the negative examples will be added to the training set.
        max_negatives (:obj:`int`, `optional`, defaults to 5):
            The maximum number of negative examples to add to the training set.
        add_with_probability (:obj:`float`, `optional`, defaults to 1.0):
            The probability with which to add the negative examples to the training set.
        refresh_every_n_epochs (:obj:`int`, `optional`, defaults to 1):
            The number of epochs after which to refresh the index.
    """

    def __init__(
        self,
        k: int = 100,
        batch_size: int = 32,
        num_workers: int = 0,
        force_reindex: bool = False,
        retriever_dir: Optional[Path] = None,
        stages: Sequence[Union[str, RunningStage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        metrics_to_monitor: List[str] = None,
        threshold: float = 0.8,
        max_negatives: int = 5,
        add_with_probability: float = 1.0,
        refresh_every_n_epochs: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(
            k=k,
            batch_size=batch_size,
            num_workers=num_workers,
            force_reindex=force_reindex,
            retriever_dir=retriever_dir,
            stages=stages,
            other_callbacks=other_callbacks,
            dataset=dataset,
            *args,
            **kwargs,
        )
        if metrics_to_monitor is None:
            metrics_to_monitor = ["val_loss"]
        self.metrics_to_monitor = metrics_to_monitor
        self.threshold = threshold
        self.max_negatives = max_negatives
        self.add_with_probability = add_with_probability
        self.refresh_every_n_epochs = refresh_every_n_epochs

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> dict:
        """
        Computes the predictions of a retriever model on a dataset and computes the negative
        examples for the training set.

        Args:
            trainer (:obj:`pl.Trainer`):
                The trainer object.
            pl_module (:obj:`pl.LightningModule`):
                The lightning module.

        Returns:
            A dictionary containing the negative examples.
        """
        stage = trainer.state.stage
        if stage not in self.stages:
            return {}

        if self.metrics_to_monitor not in trainer.logged_metrics:
            if trainer.global_rank == 0:
                logger.warning(
                    f"Metric `{self.metrics_to_monitor}` not found in trainer.logged_metrics. "
                    f"Available metrics: {trainer.logged_metrics.keys()}"
                )
            return {}

        if trainer.logged_metrics[self.metrics_to_monitor] < self.threshold:
            return {}

        if trainer.current_epoch % self.refresh_every_n_epochs != 0:
            return {}

        # if all(
        #     [
        #         trainer.logged_metrics.get(metric) is None
        #         for metric in self.metrics_to_monitor
        #     ]
        # ):
        #     raise ValueError(
        #         f"No metric from {self.metrics_to_monitor} not found in trainer.logged_metrics"
        #         f"Available metrics: {trainer.logged_metrics.keys()}"
        #     )

        # if all(
        #     [
        #         trainer.logged_metrics.get(metric) < self.threshold
        #         for metric in self.metrics_to_monitor
        #         if trainer.logged_metrics.get(metric) is not None
        #     ]
        # ):
        #     return {}

        if trainer.current_epoch % self.refresh_every_n_epochs != 0:
            return {}

        if trainer.global_rank == 0:
            logger.info(
                f"At least one metric from {self.metrics_to_monitor} is above threshold "
                f"{self.threshold}. Computing hard negatives."
            )

        # make a copy of the dataset to avoid modifying the original one
        # dataset = trainer.train_dataloader.dataset
        # dataset = GoldenRetrieverStreamingDataset(
        #     name="aida_train_hn",
        #     question_tokenizer=trainer.train_dataloader.dataset.question_tokenizer,
        #     local="/home/ric/Projects/golden-retriever/data/dpr-like/el/mosaic/train",
        #     batch_size=32,
        #     predownload=64*64,
        #     shuffle_seed=42,
        # )
        # dataset_copy = deepcopy(trainer.datamodule.train_dataset)
        # dataset = trainer.train_dataloader.dataset
        # if isinstance(self.datasets, DictConfig):
        #     logger.debug("Instantiating dataset")
        #     # OmegaConf.update(self.datasets, "batch_size", self.batch_size)
        #     # dataset = hydra.utils.instantiate(
        #     #     self.datasets,
        #     #     question_tokenizer=trainer.train_dataloader.dataset.question_tokenizer,
        #     # )
        #     # dataset = GoldenRetrieverStreamingDataset(
        #     #     **self.datasets,
        #     #     question_tokenizer=trainer.train_dataloader.dataset.question_tokenizer,
        #     #     passage_tokenizer=trainer.train_dataloader.dataset.passage_tokenizer,
        #     # )
        #     dataset_kwargs = OmegaConf.to_container(self.datasets, resolve=True)
        #     # dataset_kwargs.update({"use_cache": False})
        #     dataset, _ = trainer.datamodule.dataset_builder(
        #         name="train_hn",
        #         batch_size=self.batch_size,
        #         question_tokenizer=trainer.train_dataloader.dataset.question_tokenizer,
        #         passage_tokenizer=trainer.train_dataloader.dataset.passage_tokenizer,
        #         dataset_kwargs=dataset_kwargs,
        #     )

        # dataloader = GoldenStreamingDataLoader(
        #     dataset,
        #     collate_fn=GoldenRetrieverCollator(
        #         pad_token_type_id=dataset.question_tokenizer.pad_token_type_id,
        #     ),
        #     batch_size=dataset.batch_size,
        #     num_workers=0, #self.num_workers,
        #     # pin_memory=True,
        #     # prefetch_factor=(
        #     #     max(1, 8 * dataset.batch_size // self.num_workers)
        #     #     if self.num_workers > 0
        #     #     else None
        #     # ),
        # )
        predictions = {}
        clean_stale_shared_memory()
        dataset, _ = trainer.datamodule.dataset_builder(
            dataset=trainer.train_dataloader.dataset.original_local,
            # dataset="/home/ric/Projects/golden-retriever/data/el/aida_32_tokens_topic/train2.json",
            name="train_dataset_hn",
            batch_size=trainer.train_dataloader.dataset.batch_size,
            question_tokenizer=trainer.train_dataloader.dataset.question_tokenizer,
            passage_tokenizer=trainer.train_dataloader.dataset.passage_tokenizer,
            shuffle=True,  # force shuffle True for training
            shuffle_seed=trainer.datamodule.seed,
            dataset_kwargs=trainer.datamodule.train_dataset_kwargs,
        )
        dataloader: GoldenStreamingDataLoader = trainer.datamodule.train_dataloader(dataset)
        dataloader.load_state_dict(trainer.train_dataloader.state_dict())
        # dataset = dataloader.dataset
        # for b in dataloader:
        #     break
        predictions = super().__call__(
            trainer,
            pl_module,
            datasets=dataset,
            dataloaders=dataloader,
            limit_batches=trainer.val_check_interval, # TODO check if val_check_interval is a float and convert to a int
            *args,
            **kwargs,
        )
        logger.info(f"Computing hard negatives for epoch {trainer.current_epoch}")
        # predictions is a dict with the dataloader index as key and the predictions as value
        # since we only have one dataloader, we can get the predictions directly
        predictions = list(predictions.values())[0]
        # store the predictions in a dictionary for faster access based on the sample index
        hard_negatives_list = {}
        for prediction in tqdm(predictions, desc="Collecting hard negatives"):
            if random.random() < 1 - self.add_with_probability:
                continue
            top_k_passages = prediction["predictions"]
            gold_passages = prediction["gold"]
            # get the ids of the max_negatives wrong passages with the highest similarity
            wrong_passages = [
                passage_id
                for passage_id in top_k_passages
                if passage_id not in gold_passages
            ][: self.max_negatives]
            hard_negatives_list[prediction["sample_idx"]] = wrong_passages

        # all gather hard_negatives_list
        hard_negatives_list = dist.all_gather_object(hard_negatives_list)
        # merge list of dicts
        hard_negatives_list = {k: v for d in hard_negatives_list for k, v in d.items()}

        hn_manager: HardNegativesManagerThread | None = None
        # update on rank 0 and broadcast to other ranks
        if trainer.global_rank == 0:
            # HardNegativesManager is a singleton, so we need
            # to reset it before adding new hard negatives
            hn_manager = HardNegativesManagerThread(
                tokenizer=dataset.passage_tokenizer,
                max_length=dataset.max_passage_length,
            )
            hn_manager.reset()
            hn_manager.add(hard_negatives_list.keys(), hard_negatives_list.values())
            hn_manager.tokenize()
        trainer.strategy.broadcast(hn_manager, 0)

        # normalize predictions as in the original GoldenRetrieverPredictionCallback
        predictions = {0: predictions}
        return predictions


class CustomModelCheckpoint(ModelCheckpoint):
    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.save_checkpoint(trainer)

    def on_keyboard_interrupt(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.save_checkpoint(trainer)
