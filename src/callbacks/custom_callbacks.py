import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytorch_lightning as pl
import torch
import transformers as tr
from datasets import Dataset
from torch.utils.data import DataLoader

from callbacks.base import NLPTemplateCallback, PredictionCallback, Stage
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
        output_dir: Optional[Path] = None,
        other_callbacks: Optional[List[NLPTemplateCallback]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(other_callbacks)
        self.k = k
        self.report_intervals = report_intervals
        self.batch_size = batch_size
        self.output_dir = output_dir

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing predictions for stage {stage}")

        if stage not in [Stage.VALIDATION, Stage.TEST]:
            raise ValueError(
                f"Stage {stage} not supported, only `validation` and `test` are supported."
            )

        # get the dataloaders and datasets
        dataloaders = (
            trainer.val_dataloaders
            if stage == Stage.VALIDATION
            else trainer.test_dataloaders
        )
        data_sets = (
            trainer.datamodule.val_datasets
            if stage == Stage.VALIDATION
            else trainer.datamodule.test_datasets
        )

        # get the tokenizer
        tokenizer = trainer.datamodule.tokenizer

        dataloader_samples = {}
        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            logger.log(f"Computing context embeddings for dataloader {dataloader_idx}")
            if data_sets[dataloader_idx].contexts is not None:
                context_dataloader = DataLoader(
                    data_sets[dataloader_idx].contexts,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=lambda x: ModelInputs(
                        {
                            # this is a hack to normalize the batch structure
                            "contexts": tokenizer(x, padding=True, return_tensors="pt")
                        }
                    ),
                )
            else:
                context_dataloader = dataloader

            context_embeddings, context_index = self.compute_context_embeddings(
                pl_module.model.context_encoder, context_dataloader, pl_module.device
            )

            # now compute the question embeddings and compute the top-k accuracy
            logger.log(f"Computing predictions for dataloader {dataloader_idx}")
            samples_with_predictions = []
            for batch in dataloader:
                batch = batch.to(pl_module.device)
                model_outputs = pl_module.model(
                    batch.questions, contexts_encodings=context_embeddings
                )
                similarity = model_outputs["logits"]
                # get the top-k indices
                top_ks = torch.topk(
                    similarity, k=min(self.k, similarity.shape[-1]), dim=1
                ).indices
                # compute recall at k
                for sample_idx, top_k in enumerate(top_ks):
                    labels = batch.labels_for_metrics[sample_idx]
                    # get the positive context ids
                    gold_context_ids = [
                        " ".join(map(str, [c for c in context_ids.tolist() if c != 0]))
                        for context_ids, label in zip(batch.contexts.input_ids, labels)
                        if label == 1
                    ]
                    # get the top_k context ids
                    top_k_context_ids = [
                        context_index[context_idx] for context_idx in top_k.tolist()
                    ]

                    correct_predictions_ids = set(top_k_context_ids) & set(
                        gold_context_ids
                    )
                    wrong_predictions_ids = set(top_k_context_ids) - set(
                        gold_context_ids
                    )

                    # unnest the batch
                    sample_dict = {}
                    for k, v in batch.items():
                        if isinstance(v, ModelInputs):
                            sample_dict[k] = {
                                inner_k: inner_v[sample_idx]
                                for inner_k, inner_v in v.items()
                            }
                        elif isinstance(v, (list, tuple, torch.Tensor)):
                            sample_dict[k] = v[sample_idx]
                        else:
                            sample_dict[k] = v

                    sample_dict.update(
                        {
                            "gold_contexts": gold_context_ids,
                            "predictions": top_k_context_ids,
                            "correct_predictions": correct_predictions_ids,
                            "wrong_predictions": wrong_predictions_ids,
                        }
                    )
                    samples_with_predictions.append(sample_dict)

            dataloader_samples[dataloader_idx] = samples_with_predictions
            # write the predictions to a file inside the experiment folder
            if self.output_dir is None and trainer.logger is None:
                raise ValueError(
                    "You need to specify an output directory or a logger to save the predictions."
                )

            # save to file
            if self.output_dir is not None:
                prediction_folder = Path(self.output_dir)
            else:
                prediction_folder = Path(trainer.logger.experiment.dir) / "predictions"
                prediction_folder.mkdir(exist_ok=True)
            predictions_path = (
                prediction_folder / f"{stage.value}_dataloader_{dataloader_idx}.json"
            )
            self.save_predictions_to_file(
                samples_with_predictions, predictions_path, tokenizer
            )

        # return the predictions
        return dataloader_samples

    @staticmethod
    def save_predictions_to_file(
        samples: List[Dict],
        predictions_path: Union[str, os.PathLike],
        tokenizer: tr.PreTrainedTokenizer,
    ):
        predictions = []
        for sample in samples:
            # convert the context ids to text
            correct_predictions = [
                tokenizer.decode(
                    list(map(int, context_id.split(" "))),
                    skip_special_tokens=True,
                )
                for context_id in sample["correct_predictions"]
            ]
            wrong_predictions = [
                tokenizer.decode(
                    list(map(int, context_id.split(" "))),
                    skip_special_tokens=True,
                )
                for context_id in sample["wrong_predictions"]
            ]
            # convert top-k to text too
            top_k_contexts = [
                tokenizer.decode(
                    list(map(int, context_id.split(" "))),
                    skip_special_tokens=True,
                )
                for context_id in sample["predictions"]
            ]
            gold_contexts = [
                tokenizer.decode(
                    list(map(int, context_id.split(" "))),
                    skip_special_tokens=True,
                )
                for context_id in sample["gold_contexts"]
            ]
            predictions.append(
                {
                    "question": tokenizer.decode(
                        sample["questions"]["input_ids"],
                        skip_special_tokens=True,
                    ),
                    "gold_contexts": gold_contexts,
                    "top_k_contexts": top_k_contexts,
                    "correct_predictions": correct_predictions,
                    "wrong_predictions": wrong_predictions,
                }
            )
        predictions_path = Path(predictions_path)
        predictions_path.parent.mkdir(exist_ok=True)
        logger.log(f"Writing predictions to {predictions_path}")
        with open(predictions_path, "w") as f:
            json.dump(predictions, f, indent=2)

    @staticmethod
    @torch.no_grad()
    def compute_context_embeddings(
        context_encoder: torch.nn.Module,
        dataloader: DataLoader,
        device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, Dict[int, str]]:
        # Create empty lists to store the context embeddings and context index
        context_embeddings: List[torch.Tensor] = []
        context_index: Dict[int, str] = {}
        # Create an empty set to keep track of the contexts that have already been seen
        already_seen: Set[str] = set()
        # index to keep track of the contexts
        i: int = 0
        # Iterate through each batch in the dataloader
        for batch in dataloader:
            # Move the batch to the device
            batch = batch.to(device)
            # Compute the context embeddings
            context_outs = context_encoder(**batch.contexts)
            # Loop through each context in the batch
            for context_ids, e in zip(batch.contexts.input_ids, context_outs):
                # Clean up the context by removing any 0s
                cleaned_context = " ".join(
                    map(str, [c for c in context_ids.tolist() if c != 0])
                )
                # If the cleaned context has not been seen, add it to the empty lists and set
                if cleaned_context not in already_seen:
                    already_seen.add(cleaned_context)
                    context_embeddings.append(e)
                    context_index[i] = cleaned_context
                    i += 1
        # Stack the context embeddings into a tensor and return it along with the context index
        context_embeddings = torch.stack(context_embeddings, dim=0)
        return context_embeddings, context_index


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
                hits += len(set(sample["predictions"]) & set(sample["gold_contexts"]))
                total += len(set(sample["gold_contexts"]))

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
                    for
                    context_id in top_k_context_ids
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


class NegativeAugmentationCallback(NLPTemplateCallback):
    def __init__(
        self,
        metric_to_monitor: str = "val_loss",
        threshold: float = 0.8,
        max_negatives: int = 3,
        batch_size: int = 32,
        stages: Optional[List[Union[str, Stage]]] = [Stage.VALIDATION],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.metric_to_monitor = metric_to_monitor
        self.threshold = threshold
        self.max_negatives = max_negatives
        self.batch_size = batch_size
        self.stages = stages

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
        if stage not in self.stages:
            return

        if self.metric_to_monitor not in trainer.logged_metrics:
            raise ValueError(
                f"Metric {self.metric_to_monitor} not found in trainer.logged_metrics"
                f"Available metrics: {trainer.logged_metrics.keys()}"
            )
        if trainer.logged_metrics[self.metric_to_monitor] < self.threshold:
            return

        logger.log(
            f"Metric {self.metric_to_monitor} is above threshold {self.threshold}. Augmenting the dataset."
        )

        if stage not in [Stage.VALIDATION, Stage.TEST]:
            raise ValueError(
                f"Stage {stage} not supported, only `validation` and `test` are supported."
            )

        for dataloader_idx, samples in predictions.items():
            # create a defaultdict of defaultdicts to store the augmented contexts
            augmented_negative_contexts = defaultdict(lambda: defaultdict(list))
            for sample in samples:
                top_k_context_ids = sample["predictions"]
                gold_context_ids = sample["gold_contexts"]
                # get the ids of the max_negatives wrong contexts with highest similarity
                # wrong_context_ids = set(top_k_context_ids) - set(positive_context_ids)
                wrong_context_ids = [
                    context_id
                    for context_id in top_k_context_ids
                    if context_id not in gold_context_ids
                ][: self.max_negatives]

                # get at most max_negatives wrong contexts
                # wrong_context_ids = list(wrong_context_ids)[: self.max_negatives]
                # convert the context ids to a list of ints
                wrong_context_ids = [
                    [int(c) for c in context_id.split(" ")]
                    for context_id in wrong_context_ids
                ]
                # add the wrong contexts to the dataset sample
                sample_idx_in_dataset = sample["ids"]
                for context_id in wrong_context_ids:
                    augmented_negative_contexts[sample_idx_in_dataset][
                        "input_ids"
                    ].append(context_id)
                    augmented_negative_contexts[sample_idx_in_dataset][
                        "attention_mask"
                    ].append([1] * len(context_id))
                    augmented_negative_contexts[sample_idx_in_dataset][
                        "token_type_ids"
                    ].append([0] * len(context_id))

            dataset_dict = trainer.datamodule.train_dataset.data.to_dict()
            dataset_dict["augmented_contexts"] = []
            # add the augmented contexts to the dataset
            for sample_idx in dataset_dict["id"]:
                sample_idx = int(sample_idx)
                if sample_idx in augmented_negative_contexts:
                    dataset_dict["augmented_contexts"].append(
                        augmented_negative_contexts[sample_idx]
                    )
            # create a new dataset
            trainer.datamodule.train_dataset.data = Dataset.from_dict(dataset_dict)
