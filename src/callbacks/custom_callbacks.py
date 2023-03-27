import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from callbacks.base import PredictionCallback
from common.logging import get_console_logger
from common.model_inputs import ModelInputs
from data.datasets import BaseDataset
from models.model import GoldenRetriever

from pytorch_lightning.trainer.states import RunningStage


logger = get_console_logger()


class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: int = 100,
        batch_size: int = 32,
        num_workers: int = 0,
        use_faiss: bool = False,
        move_index_to_cpu: bool = True,
        force_reindex: bool = True,
        retriever_dir: Optional[Path] = None,
        stages: Set[Union[str, RunningStage]] = None,
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
        logger.log(f"Computing predictions for stage {stage.value}")
        if stage not in self.stages:
            raise ValueError(
                f"Stage {stage} not supported, only {self.stages} are supported"
            )

        # get the tokenizer
        tokenizer = trainer.datamodule.tokenizer

        self.datasets, self.dataloaders = self._get_datasets_and_dataloaders(
            self.datasets or datasets,
            self.dataloaders or dataloaders,
            trainer,
            dataloader_kwargs={
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "pin_memory": True,
                "shuffle": False,
            },
            collate_fn_kwargs={"tokenizer": tokenizer},
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
            logger.log(
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
            if self.retriever_dir is not None and trainer.current_epoch == 0:
                force_reindex = False
            else:
                force_reindex = self.force_reindex

            if (
                not force_reindex
                and self.retriever_dir is not None
                and stage == RunningStage.TESTING
            ):
                retriever = retriever.from_pretrained(self.retriever_dir)

            retriever.index(
                contexts,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                force_reindex=force_reindex,
                use_faiss=self.use_faiss,
                move_index_to_cpu=self.move_index_to_cpu,
            )

            pl_module_original_device = pl_module.device
            if (
                not self.use_faiss
                and self.move_index_to_cpu
                and pl_module.device.type == "cuda"
            ):
                pl_module.to("cpu")

            # now compute the question embeddings and compute the top-k accuracy
            logger.log(f"Computing predictions for dataset {current_dataset.name}")
            predictions = []
            start = time.time()
            for batch in tqdm(dataloader, desc="Retrieving contexts"):
                batch = batch.to(pl_module.device)
                # get the top-k indices
                retrieved_contexts, retrieved_indices = retriever.retrieve(
                    **batch.questions, k=self.k
                )
                # compute recall at k
                for sample_idx, retrieved_index in enumerate(retrieved_indices):
                    # get the positive contexts
                    gold_contexts = batch.positives[sample_idx]
                    # get the index of the gold contexts in the retrieved contexts
                    gold_context_indices = [
                        retriever.get_index_from_context(context)
                        for context in gold_contexts
                    ]
                    # correct predictions are the contexts that are in the top-k and are gold
                    correct_indices = set(gold_context_indices) & set(retrieved_index)
                    # wrong predictions are the contexts that are in the top-k and are not gold
                    wrong_indices = set(retrieved_index) - set(gold_context_indices)
                    # add the predictions to the list
                    prediction_output = {
                        "sample_idx": batch.sample_idx[sample_idx],
                        "gold": gold_contexts,
                        "predictions": retrieved_contexts[sample_idx],
                        "correct": [
                            retriever.get_context_from_index(i) for i in correct_indices
                        ],
                        "wrong": [
                            retriever.get_context_from_index(i) for i in wrong_indices
                        ],
                    }
                    if "id" in batch:
                        prediction_output["id"] = batch.id[sample_idx]
                    predictions.append(prediction_output)
            end = time.time()
            logger.log("Time to retrieve:", end - start)
            dataloader_predictions[dataloader_idx] = predictions

            if pl_module_original_device != pl_module.device:
                pl_module.to(pl_module_original_device)

        # return the predictions
        return dataloader_predictions

    @staticmethod
    def _get_contexts_dataloader(dataset, trainer):
        if dataset.contexts is None:
            logger.log(
                f"Contexts not found in dataset {dataset.name}, computing them from the dataloaders"
            )
            # get the contexts from the all the dataloader context ids
            contexts = set()  # set to avoid duplicates
            for batch in trainer.train_dataloader:
                contexts.update(
                    [
                        " ".join(map(str, [c for c in context_ids.tolist() if c != 0]))
                        for context_ids in batch.contexts.input_ids
                    ]
                )
            for d in trainer.val_dataloaders:
                for batch in d:
                    contexts.update(
                        [
                            " ".join(
                                map(str, [c for c in context_ids.tolist() if c != 0])
                            )
                            for context_ids in batch.contexts.input_ids
                        ]
                    )
            for d in trainer.test_dataloaders:
                for batch in d:
                    contexts.update(
                        [
                            " ".join(
                                map(str, [c for c in context_ids.tolist() if c != 0])
                            )
                            for context_ids in batch.contexts.input_ids
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
        metric_to_monitor: str = "val_loss",
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
        self.metric_to_monitor = metric_to_monitor
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

        if self.metric_to_monitor not in trainer.logged_metrics:
            raise ValueError(
                f"Metric {self.metric_to_monitor} not found in trainer.logged_metrics"
                f"Available metrics: {trainer.logged_metrics.keys()}"
            )
        if trainer.logged_metrics[self.metric_to_monitor] < self.threshold:
            return {}

        if trainer.current_epoch % self.refresh_every_n_epochs != 0:
            return {}

        logger.log(
            f"Metric {self.metric_to_monitor} is above threshold {self.threshold}. Computing hard negatives."
        )

        predictions = super().__call__(
            trainer,
            pl_module,
            datasets=trainer.datamodule.train_dataset,
            dataloaders=trainer.datamodule.train_dataloader(),
            *args,
            **kwargs,
        )
        # predictions is a dict with the dataloader index as key and the predictions as value
        # since we only have one dataloader, we can get the predictions directly
        predictions = list(predictions.values())[0]
        # store the predictions in a dictionary for faster access based on the sample index
        update_dict = defaultdict(lambda: defaultdict(list))
        for prediction in predictions:
            top_k_contexts = prediction["predictions"]
            gold_contexts = prediction["gold"]
            # get the ids of the max_negatives wrong contexts with the highest similarity
            wrong_contexts = [
                context_id
                for context_id in top_k_contexts
                if context_id not in gold_contexts
            ][: self.max_negatives]
            wrong_contexts_ids = trainer.datamodule.tokenizer(
                wrong_contexts,
                max_length=trainer.datamodule.train_dataset.max_context_length,
                truncation=True,
            )
            retrieved_hard_negatives = []
            for c_index in range(len(wrong_contexts)):
                p_dict = {
                    "input_ids": wrong_contexts_ids["input_ids"][c_index],
                    "attention_mask": wrong_contexts_ids["attention_mask"][c_index],
                }
                if "token_type_ids" in wrong_contexts_ids:
                    p_dict["token_type_ids"] = wrong_contexts_ids["token_type_ids"][
                        c_index
                    ]
                retrieved_hard_negatives.append(p_dict)
            update_dict[prediction["sample_idx"]][
                "retrieved_hard_negatives"
            ] = retrieved_hard_negatives
        logger.log(f"Adding hard negatives to the dataset.")
        trainer.datamodule.train_dataset.add_fields_to_samples(update_dict)

        return predictions
