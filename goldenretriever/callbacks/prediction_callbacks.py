from copy import deepcopy
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Set, Union

import psutil
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers as tr

from goldenretriever.callbacks.base import PredictionCallback
from goldenretriever.common.log import get_console_logger, get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.datasets import BaseDataset
from goldenretriever.data.dpr.hard_negatives_manager import HardNegativeManager
from goldenretriever.models.model import GoldenRetriever

console_logger = get_console_logger()
logger = get_logger(__name__, level=logging.INFO)


class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        use_faiss: bool = False,
        move_index_to_cpu: bool = True,
        precision: Union[str, int] = 32,
        index_precision: Union[str, int] = 32,
        force_reindex: bool = True,
        retriever_dir: Optional[Path] = None,
        stages: Optional[Set[Union[str, RunningStage]]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        dataloader: Optional[DataLoader] = None,
        *args,
        **kwargs,
    ):
        super().__init__(batch_size, stages, other_callbacks, dataset, dataloader)
        self.k = k
        self.num_workers = num_workers
        self.use_faiss = use_faiss
        self.move_index_to_cpu = move_index_to_cpu
        self.precision = precision
        self.index_precision = index_precision
        self.force_reindex = force_reindex
        self.retriever_dir = retriever_dir

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        datasets: Optional[
            Union[DictConfig, BaseDataset, List[DictConfig], List[BaseDataset]]
        ] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        *args,
        **kwargs,
    ) -> dict:
        stage = trainer.state.stage
        logger.info(f"Computing predictions for stage {stage.value}")
        if stage not in self.stages:
            raise ValueError(
                f"Stage `{stage}` not supported, only {self.stages} are supported"
            )

        # get the tokenizer
        tokenizer = trainer.datamodule.tokenizer

        # if datasets is not None or dataloaders is not None:
        #     self.datasets = datasets
        #     self.dataloaders = dataloaders

        self.datasets, self.dataloaders = self._get_datasets_and_dataloaders(
            datasets,
            dataloaders,
            trainer,
            dataloader_kwargs=dict(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
            ),
            collate_fn_kwargs=dict(tokenizer=tokenizer),
        )

        # set the model to eval mode
        pl_module.eval()
        # get the retriever
        retriever: GoldenRetriever = pl_module.model

        # here we will store the samples with predictions for each dataloader
        dataloader_predictions = {}
        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(self.dataloaders):
            current_dataset: BaseDataset = self.datasets[dataloader_idx]
            logger.info(
                f"Computing context embeddings for dataset {current_dataset.name}"
            )
            contexts = self._get_contexts_dataloader(current_dataset, trainer)

            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    truncation=True,
                    padding=True,
                    max_length=current_dataset.max_context_length,
                    return_tensors="pt",
                )
            )

            # check if we need to reindex the contexts and
            # also if we need to load the retriever from disk
            if (self.retriever_dir is not None and trainer.current_epoch == 0) or (
                self.retriever_dir is not None and stage == RunningStage.TESTING
            ):
                force_reindex = False
            else:
                force_reindex = self.force_reindex

            if (
                not force_reindex
                and self.retriever_dir is not None
                and stage == RunningStage.TESTING
            ):
                retriever = retriever.from_pretrained(self.retriever_dir)
                # set the retriever to eval mode if we are loading it from disk

            # you never know :)
            retriever.eval()

            retriever.index(
                contexts,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                force_reindex=force_reindex,
                use_faiss=self.use_faiss,
                move_index_to_cpu=self.move_index_to_cpu,
                precision=self.precision,
                index_precision=self.index_precision,
            )

            pl_module_original_device = pl_module.device
            if (
                not self.use_faiss
                and self.move_index_to_cpu
                and pl_module.device.type == "cuda"
            ):
                pl_module.to("cpu")

            # now compute the question embeddings and compute the top-k accuracy
            # logger.info(f"Computing predictions for dataset {current_dataset.name}")
            predictions = []
            start = time.time()
            for batch in tqdm(
                dataloader,
                desc=f"Computing predictions for dataset {current_dataset.name}",
            ):
                # batch = batch
                batch = ModelInputs(**batch).to(pl_module.device)
                # get the top-k indices
                retriever_output = retriever.retrieve(
                    **batch.questions, k=self.k, precision=self.precision
                )
                # compute recall at k
                for batch_idx, retrieved_samples in enumerate(retriever_output):
                    # get the positive contexts
                    gold_contexts = batch["positives"][batch_idx]
                    # get the index of the gold contexts in the retrieved contexts
                    gold_context_indices = [
                        retriever.get_index_from_context(context)
                        for context in gold_contexts
                    ]
                    retrieved_indices = [r.index for r in retrieved_samples]
                    retrieved_contexts = [r.label for r in retrieved_samples]
                    retrieved_scores = [r.score for r in retrieved_samples]
                    # correct predictions are the contexts that are in the top-k and are gold
                    correct_indices = set(gold_context_indices) & set(retrieved_indices)
                    # wrong predictions are the contexts that are in the top-k and are not gold
                    wrong_indices = set(retrieved_indices) - set(gold_context_indices)
                    # add the predictions to the list
                    prediction_output = dict(
                        sample_idx=batch.sample_idx[batch_idx].item(),
                        gold=gold_contexts,
                        predictions=retrieved_contexts,
                        scores=retrieved_scores,
                        correct=[
                            retriever.get_context_from_index(i) for i in correct_indices
                        ],
                        wrong=[
                            retriever.get_context_from_index(i) for i in wrong_indices
                        ],
                    )
                    predictions.append(prediction_output)
            end = time.time()
            logger.info(f"Time to retrieve: {str(end - start)}")

            dataloader_predictions[dataloader_idx] = predictions

            if pl_module_original_device != pl_module.device:
                pl_module.to(pl_module_original_device)

        # return the predictions
        return dataloader_predictions

    @staticmethod
    def _get_contexts_dataloader(dataset, trainer):
        if dataset.contexts is None:
            logger.info(
                f"Contexts not found in dataset {dataset.name}, computing them from the dataloaders"
            )
            # get the contexts from the all the dataloader context ids
            contexts = set()  # set to avoid duplicates
            for batch in trainer.train_dataloader:
                contexts.update(
                    [
                        " ".join(map(str, [c for c in context_ids.tolist() if c != 0]))
                        for context_ids in batch["contexts"]["input_ids"]
                    ]
                )
            for d in trainer.val_dataloaders:
                for batch in d:
                    contexts.update(
                        [
                            " ".join(
                                map(str, [c for c in context_ids.tolist() if c != 0])
                            )
                            for context_ids in batch["contexts"]["input_ids"]
                        ]
                    )
            for d in trainer.test_dataloaders:
                for batch in d:
                    contexts.update(
                        [
                            " ".join(
                                map(str, [c for c in context_ids.tolist() if c != 0])
                            )
                            for context_ids in batch["contexts"]["input_ids"]
                        ]
                    )
            contexts = list(contexts)
        else:
            contexts = dataset.contexts
        return contexts


class NegativeAugmentationCallback(GoldenRetrieverPredictionCallback):
    def __init__(
        self,
        k: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        use_faiss: bool = False,
        move_index_to_cpu: bool = False,
        force_reindex: bool = False,
        retriever_dir: Optional[Path] = None,
        stages: Set[Union[str, RunningStage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        metrics_to_monitor: List[str] = None,
        threshold: float = 0.8,
        max_negatives: int = 5,
        refresh_every_n_epochs: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(
            k=k,
            batch_size=batch_size,
            num_workers=num_workers,
            use_faiss=use_faiss,
            move_index_to_cpu=move_index_to_cpu,
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
        self.refresh_every_n_epochs = refresh_every_n_epochs

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> dict:

        stage = trainer.state.stage
        if stage not in self.stages:
            return {}

        if self.metrics_to_monitor not in trainer.logged_metrics:
            raise ValueError(
                f"Metric `{self.metrics_to_monitor}` not found in trainer.logged_metrics"
                f"Available metrics: {trainer.logged_metrics.keys()}"
            )
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

        logger.info(
            f"At least one metric from {self.metrics_to_monitor} is above threshold "
            f"{self.threshold}. Computing hard negatives."
        )

        # reset hard_negatives_manager to avoid memory leaks
        trainer.datamodule.train_dataset.hard_negatives_manager = None
        # make a copy of the dataset to avoid modifying the original one
        dataset_copy = deepcopy(trainer.datamodule.train_dataset)
        predictions = super().__call__(
            trainer,
            pl_module,
            datasets=dataset_copy,
            dataloaders=DataLoader(
                dataset_copy.to_torch_dataset(),
                shuffle=False,
                batch_size=None,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=lambda x: x,
            ),
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
            top_k_contexts = prediction["predictions"]
            gold_contexts = prediction["gold"]
            # get the ids of the max_negatives wrong contexts with the highest similarity
            wrong_contexts = [
                context_id
                for context_id in top_k_contexts
                if context_id not in gold_contexts
            ][: self.max_negatives]
            hard_negatives_list[prediction["sample_idx"]] = wrong_contexts

        hn_manager = HardNegativeManager(
            tokenizer=trainer.datamodule.tokenizer,
            max_length=trainer.datamodule.train_dataset.max_context_length,
            data=hard_negatives_list,
        )
        trainer.datamodule.train_dataset.hard_negatives_manager = hn_manager

        # normalize predictions as in the original GoldenRetrieverPredictionCallback
        predictions = {0: predictions}
        return predictions
