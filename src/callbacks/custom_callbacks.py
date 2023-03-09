import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import transformers as tr
import datasets
from datasets import Dataset
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from callbacks.base import NLPTemplateCallback, PredictionCallback, Stage
from data.datasets import BaseDataset
from utils.logging import get_console_logger
from utils.model_inputs import ModelInputs

# from faiss.indexer import FaissIndexer

logger = get_console_logger()


class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: int = 100,
        report_intervals: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        output_dir: Optional[Path] = None,
        stages: Set[Union[str, Stage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(stages, other_callbacks)
        self.k = k
        self.report_intervals = report_intervals
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.dataset = dataset

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing predictions for stage {stage.value}")

        if stage not in self.stages:
            raise ValueError(
                f"Stage {stage} not supported, only {self.stages} are supported"
            )

        # get the tokenizer
        tokenizer = trainer.datamodule.tokenizer

        # if a dataset is provided, use it
        if self.dataset is not None:
            # get dataset
            if isinstance(self.dataset, DictConfig):
                self.dataset = hydra.utils.instantiate(self.dataset, _recursive_=False)
            datasets = [self.dataset]
            dataloaders = [
                DataLoader(
                    datasets[0],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=partial(datasets[0].collate_fn, tokenizer=tokenizer),
                )
            ]
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

        # set the model to eval mode
        pl_module.eval()
        # get the retriever
        retriever = pl_module.model

        # here we will store the samples with predictions for each dataloader
        dataloader_predictions = {}
        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            logger.log(
                f"Computing context embeddings for dataset {datasets[dataloader_idx].name}"
            )
            if datasets[dataloader_idx].contexts is None:
                logger.log(
                    f"Contexts not found in dataset {datasets[dataloader_idx].name}, computing them from the dataloaders"
                )
                # get the contexts from the all the dataloader context ids
                contexts = set()  # set to avoid duplicates
                for batch in trainer.train_dataloader:
                    contexts.update(
                        [
                            " ".join(
                                map(str, [c for c in context_ids.tolist() if c != 0])
                            )
                            for context_ids in batch.contexts.input_ids
                        ]
                    )
                for d in trainer.val_dataloaders:
                    for batch in d:
                        contexts.update(
                            [
                                " ".join(
                                    map(
                                        str, [c for c in context_ids.tolist() if c != 0]
                                    )
                                )
                                for context_ids in batch.contexts.input_ids
                            ]
                        )
                for d in trainer.test_dataloaders:
                    for batch in d:
                        contexts.update(
                            [
                                " ".join(
                                    map(
                                        str, [c for c in context_ids.tolist() if c != 0]
                                    )
                                )
                                for context_ids in batch.contexts.input_ids
                            ]
                        )
                contexts = list(contexts)
            else:
                contexts = datasets[dataloader_idx].contexts

            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    truncation=True,
                    padding=True,
                    max_length=datasets[dataloader_idx].max_context_length,
                    return_tensors="pt",
                )
            )
            retriever.index(
                contexts,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                force_reindex=True,
            )

            # now compute the question embeddings and compute the top-k accuracy
            logger.log(
                f"Computing predictions for dataset {datasets[dataloader_idx].name}"
            )
            predictions = []
            for batch in dataloader:
                batch = batch.to(pl_module.device)
                # get the top-k indices
                retrieved_contexts, retrieved_indices = retriever.retrieve(
                    **batch.questions, k=self.k
                )
                # compute recall at k
                for sample_idx, retrieved_index in enumerate(retrieved_indices):
                    labels = batch.labels_for_metrics[sample_idx]
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
                    predictions.append(
                        {
                            "gold": gold_contexts,
                            "predictions": retrieved_contexts[sample_idx],
                            "correct": [
                                retriever.get_context_from_index(i)
                                for i in correct_indices
                            ],
                            "wrong": [
                                retriever.get_context_from_index(i)
                                for i in wrong_indices
                            ],
                        }
                    )
            dataloader_predictions[dataloader_idx] = predictions

            # update the dataset with the predictions
            datasets[dataloader_idx].update_data(
                "gold", [p["gold"] for p in predictions]
            )
            datasets[dataloader_idx].update_data(
                "predictions", [p["predictions"] for p in predictions]
            )
            datasets[dataloader_idx].update_data(
                "correct", [p["correct"] for p in predictions]
            )
            datasets[dataloader_idx].update_data(
                "wrong", [p["wrong"] for p in predictions]
            )

            # write the predictions to a file inside the experiment folder
            if self.output_dir is None and trainer.logger is None:
                logger.log(
                    "You need to specify an output directory or a logger to save the predictions."
                )
                # save to file
                if self.output_dir is not None:
                    prediction_folder = Path(self.output_dir)
                else:
                    prediction_folder = (
                        Path(trainer.logger.experiment.dir) / "predictions"
                    )
                    prediction_folder.mkdir(exist_ok=True)
                predictions_path = (
                    prediction_folder
                    / f"{datasets[dataloader_idx].name}_{dataloader_idx}.json"
                )
                datasets[dataloader_idx].save_samples(predictions_path)

        # return the predictions
        return dataloader_predictions


class NegativeAugmentationCallback(GoldenRetrieverPredictionCallback):
    def __init__(
        self,
        k: int = 100,
        report_intervals: Optional[int] = None,
        metric_to_monitor: str = "val_loss",
        threshold: float = 0.8,
        max_negatives: int = 3,
        batch_size: int = 32,
        output_dir: Optional[Path] = None,
        stages: Set[Union[str, Stage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[DictConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            k=k,
            report_intervals=report_intervals,
            batch_size=batch_size,
            output_dir=output_dir,
            stages=stages,
            other_callbacks=other_callbacks,
            dataset=dataset,
            *args,
            **kwargs,
        )
        self.metric_to_monitor = metric_to_monitor
        self.threshold = threshold
        self.max_negatives = max_negatives

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        *args,
        **kwargs,
    ) -> dict:
        if stage not in self.stages:
            return {}

        if self.metric_to_monitor not in trainer.logged_metrics:
            raise ValueError(
                f"Metric {self.metric_to_monitor} not found in trainer.logged_metrics"
                f"Available metrics: {trainer.logged_metrics.keys()}"
            )
        if trainer.logged_metrics[self.metric_to_monitor] < self.threshold:
            return {}

        logger.log(
            f"Metric {self.metric_to_monitor} is above threshold {self.threshold}. Augmenting the dataset."
        )

        predictions = super().__call__(trainer, pl_module, stage, *args, **kwargs)
        for dataloader_idx, samples in predictions.items():
            # create a defaultdict of defaultdicts to store the augmented contexts
            augmented_negative_contexts = defaultdict(lambda: defaultdict(list))
            for sample in samples:
                top_k_contexts = sample["predictions"]
                gold_contexts = sample["gold"]
                # get the ids of the max_negatives wrong contexts with highest similarity
                wrong_contexts = [
                    context_id
                    for context_id in top_k_contexts
                    if context_id not in gold_contexts
                ][: self.max_negatives]
                # add the wrong contexts to the dataset sample
                sample_idx_in_dataset = sample["ids"]
                wrong_contexts_ids = trainer.datamodule.tokenizer(
                    wrong_contexts,
                    max_length=trainer.datamodule.train_dataset.max_context_length,
                    truncation=True,
                )
                for c_index in range(len(wrong_contexts)):
                    augmented_negative_contexts[sample_idx_in_dataset][
                        "input_ids"
                    ].append(wrong_contexts_ids["input_ids"][c_index])
                    augmented_negative_contexts[sample_idx_in_dataset][
                        "attention_mask"
                    ].append(wrong_contexts_ids["attention_mask"][c_index])
                    if "token_type_ids" in wrong_contexts_ids:
                        augmented_negative_contexts[sample_idx_in_dataset][
                            "token_type_ids"
                        ].append(wrong_contexts_ids["token_type_ids"][c_index])

            # dataset_dict = trainer.datamodule.train_dataset.data.to_dict()
            # dataset_dict["augmented_contexts"] = []
            # # add the augmented contexts to the dataset
            # for sample_idx in dataset_dict["id"]:
            #     if sample_idx in augmented_negative_contexts:
            #         dataset_dict["augmented_contexts"].append(
            #             augmented_negative_contexts[sample_idx]
            #         )
            # # create a new dataset
            # trainer.datamodule.train_dataset.data = Dataset.from_dict(dataset_dict)

            # order augmented_negative_contexts by sample_idx_in_dataset and get the values
            augmented_negative_contexts = [
                augmented_negative_contexts[i]
                for i in sorted(augmented_negative_contexts.keys())
            ]
            trainer.datamodule.train_dataset.data[dataloader_idx].update_data(
                "augmented_contexts", augmented_negative_contexts
            )

        return predictions


class TopKEvaluationCallback(NLPTemplateCallback):
    def __init__(
        self,
        k: int = 100,
        report_intervals: Optional[int] = None,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.report_intervals = report_intervals
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        predictions: Dict,
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing recall@{self.k}")

        # metrics to return
        metrics = {}

        if stage not in [Stage.VALIDATION, Stage.TEST]:
            raise ValueError(
                f"Stage {stage} not supported, only `validation` and `test` are supported."
            )

        for dataloader_idx, samples in predictions.items():
            hits, total = 0, 0
            for sample in samples:
                # compute the recall at k
                hits += len(set(sample["predictions"]) & set(sample["gold"]))
                total += len(set(sample["gold"]))

            # compute the mean recall at k
            recall_at_k = hits / total
            metrics[f"recall@{self.k}_{dataloader_idx}"] = recall_at_k
        metrics[f"recall@{self.k}"] = sum(metrics.values()) / len(metrics)

        metrics = {f"{stage.value}_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics


class NYTTopKEvaluationCallback(TopKEvaluationCallback):
    def __init__(
        self,
        label_mapping: Dict[str, List[str]],
        k: int = 100,
        report_intervals: Optional[int] = None,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(k, report_intervals, batch_size, *args, **kwargs)
        self.label_mapping = label_mapping

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        predictions: Dict,
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing recall@{self.k}")

        # metrics to return
        metrics = {}

        if stage not in [Stage.VALIDATION, Stage.TEST]:
            raise ValueError(
                f"Stage {stage} not supported, only `validation` and `test` are supported."
            )

        for dataloader_idx, samples in predictions.items():
            # shitty hack to get the label mapping normalized to the contexts
            label_mapping = {
                label: [
                    " ".join(
                        map(
                            str,
                            [
                                c
                                for c in trainer.datamodule.tokenizer(description)[
                                    "input_ids"
                                ]
                                if c != 0
                            ],
                        )
                    )
                    for description in descriptions
                ]
                for label, descriptions in self.label_mapping.items()
            }
            # invert the label mapping
            inverted_label_mapping = {
                description: label
                for label, descriptions in label_mapping.items()
                for description in descriptions
            }
            # now compute the question embeddings and compute the top-k accuracy
            logger.log(f"Computing recall@{self.k} for dataloader {dataloader_idx}")
            hits, total = 0, 0
            for sample in samples:
                gold_contexts_ids = sample["gold_contexts"]
                gold_labels = [
                    label
                    for label, descriptions in label_mapping.items()
                    if set(descriptions) & set(gold_contexts_ids)
                ]
                # get the top_k context ids
                top_k_context_ids = sample["predictions"]
                top_k_labels = [
                    inverted_label_mapping[context_id]
                    for context_id in top_k_context_ids
                ]
                # remove duplicates and preserve the order
                top_k_labels = list(dict.fromkeys(top_k_labels))
                top_k_labels = top_k_labels[: self.k]
                # compute
                hits += len(set(top_k_labels) & set(gold_labels))
                total += len(set(gold_labels))

            # compute the mean recall at k
            recall_at_k = hits / total
            metrics[f"recall@{self.k}_{dataloader_idx}"] = recall_at_k
        metrics[f"recall@{self.k}"] = sum(metrics.values()) / len(metrics)

        metrics = {f"{stage.value}_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics
