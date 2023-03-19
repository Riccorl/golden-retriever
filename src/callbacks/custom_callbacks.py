import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple

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


class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        use_faiss: bool = False,
        force_reindex: bool = True,
        output_dir: Optional[Path] = None,
        save_predictions: bool = True,
        stages: Set[Union[str, Stage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(stages, other_callbacks)
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_faiss = use_faiss
        self.force_reindex = force_reindex
        self.output_dir = output_dir
        self.save_predictions = save_predictions
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

        datasets, dataloaders = self._get_datasets_and_dataloaders(
            self.dataset, self.batch_size, self.num_workers, stage, trainer, tokenizer
        )

        # set the model to eval mode
        pl_module.eval()
        # get the retriever
        retriever: GoldenRetriever = pl_module.model

        # here we will store the samples with predictions for each dataloader
        dataloader_predictions = {}
        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            logger.log(
                f"Computing context embeddings for dataset {datasets[dataloader_idx].name}"
            )
            contexts = self._get_contexts_dataloader(datasets[dataloader_idx], trainer)

            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    truncation=True,
                    padding=True,
                    max_length=datasets[dataloader_idx].max_context_length,
                    return_tensors="pt",
                )
            )
            use_gpu = (pl_module.device.type == "cuda") if not self.use_faiss else False
            retriever.index(
                contexts,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                force_reindex=self.force_reindex,
                use_faiss=self.use_faiss,
                use_gpu=use_gpu,  # (pl_module.device.type == "cuda"),
            )

            # now compute the question embeddings and compute the top-k accuracy
            logger.log(
                f"Computing predictions for dataset {datasets[dataloader_idx].name}"
            )
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
                    predictions.append(
                        {
                            "id": batch.id[sample_idx],
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
            end = time.time()
            logger.log("Time to retrieve:", end - start)
            dataloader_predictions[dataloader_idx] = predictions

            # update the dataset with the predictions
            new_names = ["gold", "predictions", "correct", "wrong"]
            new_columns = [[p[name] for p in predictions] for name in new_names]
            datasets[dataloader_idx].update_data(new_names, new_columns, overwrite=True)

            # write the predictions to a file inside the experiment folder
            if self.output_dir is None and trainer.logger is None:
                logger.log(
                    "You need to specify an output directory or a logger to save the predictions."
                )
            else:
                # save to file
                if self.save_predictions:
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
                    logger.log(f"Saving predictions to {predictions_path}")
                    datasets[dataloader_idx].save_data(
                        predictions_path,
                        remove_columns=[
                            "context",
                            "positives",
                            "negatives",
                            "wrong",
                            "positive_ctxs",
                            "negative_ctxs",
                            "hard_negative_ctxs",
                            "positive_index_end",
                        ],
                    )

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

    @staticmethod
    def _get_datasets_and_dataloaders(
        dataset: Optional[Union[Dataset, DictConfig]],
        batch_size: int,
        num_workers: int,
        stage: Stage,
        trainer: pl.Trainer,
        tokenizer: tr.PreTrainedTokenizer,
    ) -> Tuple[List[DPRDataset], List[DataLoader]]:
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
            # get dataset
            if isinstance(dataset, DictConfig):
                dataset = hydra.utils.instantiate(dataset, _recursive_=False)
            datasets = [dataset]
            dataloaders = [
                DataLoader(
                    datasets[0],
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
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
        return datasets, dataloaders


class NegativeAugmentationCallback(GoldenRetrieverPredictionCallback):
    def __init__(
        self,
        k: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        force_reindex: bool = False,
        use_faiss: bool = False,
        output_dir: Optional[Path] = None,
        save_predictions: bool = False,
        stages: Set[Union[str, Stage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        metric_to_monitor: str = "val_loss",
        threshold: float = 0.8,
        max_negatives: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(
            k=k,
            batch_size=batch_size,
            num_workers=num_workers,
            force_reindex=force_reindex,
            use_faiss=use_faiss,
            output_dir=output_dir,
            save_predictions=save_predictions,
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
            f"Metric {self.metric_to_monitor} is above threshold {self.threshold}. Computing hard negatives."
        )

        predictions = super().__call__(trainer, pl_module, stage, *args, **kwargs)
        for _, samples in predictions.items():
            # create a defaultdict of defaultdicts to store the augmented contexts
            # retrieved_hard_negatives = defaultdict(lambda: defaultdict(list))
            retrieved_hard_negatives = []
            for s_idx, sample in enumerate(samples):
                top_k_contexts = sample["predictions"]
                gold_contexts = sample["gold"]
                # get the ids of the max_negatives wrong contexts with the highest similarity
                wrong_contexts = [
                    context_id
                    for context_id in top_k_contexts
                    if context_id not in gold_contexts
                ][: self.max_negatives]
                # add the wrong contexts to the dataset sample
                # sample_idx_in_dataset = sample["id"]
                wrong_contexts_ids = trainer.datamodule.tokenizer(
                    wrong_contexts,
                    max_length=trainer.datamodule.train_dataset.max_context_length,
                    truncation=True,
                )
                retrieved_hard_negatives.append(
                    [
                        {
                            "input_ids": wrong_contexts_ids["input_ids"][c_index],
                            "attention_mask": wrong_contexts_ids["attention_mask"][
                                c_index
                            ],
                            "token_type_ids": wrong_contexts_ids["token_type_ids"][
                                c_index
                            ],
                        }
                        for c_index in range(len(wrong_contexts))
                    ]
                )
            trainer.datamodule.train_dataset.update_data(
                "retrieved_hard_negatives", retrieved_hard_negatives, overwrite=True
            )

        return predictions


class TopKEvaluationCallback(NLPTemplateCallback):
    def __init__(
        self,
        k: int = 100,
        prefix: Optional[str] = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.prefix = prefix
        self.verbose = verbose

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
        if self.verbose:
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
                # cut the predictions to the first k elements
                predictions = sample["predictions"][: self.k]
                hits += len(set(predictions) & set(sample["gold"]))
                total += len(set(sample["gold"]))

            # compute the mean recall at k
            recall_at_k = hits / total
            metrics[f"recall@{self.k}_{dataloader_idx}"] = recall_at_k
        metrics[f"recall@{self.k}"] = sum(metrics.values()) / len(metrics)

        if self.prefix is not None:
            metrics = {f"{self.prefix}_{k}": v for k, v in metrics.items()}
        else:
            metrics = {f"{stage.value}_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        if self.verbose:
            logger.log(
                f"Recall@{self.k} on {stage.value}: {metrics[f'{stage.value}_recall@{self.k}']}"
            )

        return metrics


class NYTTopKEvaluationCallback(TopKEvaluationCallback):
    def __init__(
        self,
        label_mapping: Dict[str, List[str]],
        k: int = 100,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(k, batch_size, *args, **kwargs)
        self.label_mapping = label_mapping
        # invert the label mapping
        self.inverted_label_mapping = {
            description: label
            for label, descriptions in label_mapping.items()
            for description in descriptions
        }

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
            # now compute the question embeddings and compute the top-k accuracy
            logger.log(f"Computing recall@{self.k} for dataloader {dataloader_idx}")
            hits, total = 0, 0
            for sample in samples:
                gold_contexts = sample["gold"]
                gold_labels = [
                    label
                    for label, descriptions in self.label_mapping.items()
                    if set(descriptions) & set(gold_contexts)
                ]
                # get the top_k context ids
                top_k_contexts = sample["predictions"]
                top_k_labels = [
                    self.inverted_label_mapping[context_id]
                    for context_id in top_k_contexts
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
