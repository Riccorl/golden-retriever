import logging
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Set, Union

import lightning as pl
import torch
from omegaconf import DictConfig
from lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from tqdm import tqdm

from goldenretriever.callbacks.base import PredictionCallback
from goldenretriever.common.log import get_console_logger, get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.base.datasets import BaseDataset
from goldenretriever.data.datasets import GoldenRetrieverDataset
from goldenretriever.data.utils import HardNegativesManager
from goldenretriever.retriever.golden_retriever import GoldenRetriever
from goldenretriever.retriever.indexers.base import BaseDocumentIndex

console_logger = get_console_logger()
logger = get_logger(__name__, level=logging.INFO)


class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        document_index: Optional[BaseDocumentIndex] = None,
        precision: Union[str, int] = 32,
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
        self.document_index = document_index
        self.precision = precision
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
        )

        # set the model to eval mode
        pl_module.eval()
        # get the retriever
        retriever: GoldenRetriever = pl_module.model

        # here we will store the samples with predictions for each dataloader
        dataloader_predictions = {}
        # compute the passage embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(self.dataloaders):
            current_dataset: GoldenRetrieverDataset = self.datasets[dataloader_idx]
            logger.info(
                f"Computing passage embeddings for dataset {current_dataset.name}"
            )
            # passages = self._get_passages_dataloader(current_dataset, trainer)

            tokenizer = current_dataset.tokenizer
            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    truncation=True,
                    padding=True,
                    max_length=current_dataset.max_passage_length,
                    return_tensors="pt",
                )
            )

            # check if we need to reindex the passages and
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
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                max_length=current_dataset.max_passage_length,
                collate_fn=collate_fn,
                precision=self.precision,
                compute_on_cpu=False,
                force_reindex=force_reindex,
            )

            # pl_module_original_device = pl_module.device
            # if (
            #     and pl_module.device.type == "cuda"
            # ):
            #     pl_module.to("cpu")

            # now compute the question embeddings and compute the top-k accuracy
            predictions = []
            start = time.time()
            for batch in tqdm(
                dataloader,
                desc=f"Computing predictions for dataset {current_dataset.name}",
            ):
                batch = batch.to(pl_module.device)
                # get the top-k indices
                retriever_output = retriever.retrieve(
                    **batch.questions, k=self.k, precision=self.precision
                )
                # compute recall at k
                for batch_idx, retrieved_samples in enumerate(retriever_output):
                    # get the positive passages
                    gold_passages = batch["positives"][batch_idx]
                    # get the index of the gold passages in the retrieved passages
                    gold_passage_indices = [
                        retriever.get_index_from_passage(passage)
                        for passage in gold_passages
                    ]
                    retrieved_indices = [r.index for r in retrieved_samples]
                    retrieved_passages = [r.label for r in retrieved_samples]
                    retrieved_scores = [r.score for r in retrieved_samples]
                    # correct predictions are the passages that are in the top-k and are gold
                    correct_indices = set(gold_passage_indices) & set(retrieved_indices)
                    # wrong predictions are the passages that are in the top-k and are not gold
                    wrong_indices = set(retrieved_indices) - set(gold_passage_indices)
                    # add the predictions to the list
                    prediction_output = dict(
                        sample_idx=batch.sample_idx[batch_idx],
                        gold=gold_passages,
                        predictions=retrieved_passages,
                        scores=retrieved_scores,
                        correct=[
                            retriever.get_passage_from_index(i) for i in correct_indices
                        ],
                        wrong=[
                            retriever.get_passage_from_index(i) for i in wrong_indices
                        ],
                    )
                    predictions.append(prediction_output)
            end = time.time()
            logger.info(f"Time to retrieve: {str(end - start)}")

            dataloader_predictions[dataloader_idx] = predictions

            # if pl_module_original_device != pl_module.device:
            #     pl_module.to(pl_module_original_device)

        # return the predictions
        return dataloader_predictions

    # @staticmethod
    # def _get_passages_dataloader(
    #     indexer: Optional[BaseIndexer] = None,
    #     dataset: Optional[GoldenRetrieverDataset] = None,
    #     trainer: Optional[pl.Trainer] = None,
    # ):
    #     if indexer is None:
    #         logger.info(
    #             f"Indexer is None, creating indexer from passages not found in dataset {dataset.name}, computing them from the dataloaders"
    #         )
    #         # get the passages from the all the dataloader passage ids
    #         passages = set()  # set to avoid duplicates
    #         for batch in trainer.train_dataloader:
    #             passages.update(
    #                 [
    #                     " ".join(map(str, [c for c in passage_ids.tolist() if c != 0]))
    #                     for passage_ids in batch["passages"]["input_ids"]
    #                 ]
    #             )
    #         for d in trainer.val_dataloaders:
    #             for batch in d:
    #                 passages.update(
    #                     [
    #                         " ".join(
    #                             map(str, [c for c in passage_ids.tolist() if c != 0])
    #                         )
    #                         for passage_ids in batch["passages"]["input_ids"]
    #                     ]
    #                 )
    #         for d in trainer.test_dataloaders:
    #             for batch in d:
    #                 passages.update(
    #                     [
    #                         " ".join(
    #                             map(str, [c for c in passage_ids.tolist() if c != 0])
    #                         )
    #                         for passage_ids in batch["passages"]["input_ids"]
    #                     ]
    #                 )
    #         passages = list(passages)
    #     else:
    #         passages = dataset.passages
    #     return passages


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
        stages: Set[Union[str, RunningStage]] = None,
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

        # make a copy of the dataset to avoid modifying the original one
        trainer.datamodule.train_dataset.hn_manager = None
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

        trainer.datamodule.train_dataset.hn_manager = HardNegativesManager(
            tokenizer=trainer.datamodule.train_dataset.tokenizer,
            max_length=trainer.datamodule.train_dataset.max_passage_length,
            data=hard_negatives_list,
        )

        # normalize predictions as in the original GoldenRetrieverPredictionCallback
        predictions = {0: predictions}
        return predictions
