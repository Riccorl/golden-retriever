from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
import torch

from callbacks.base import NLPTemplateCallback
from common.logging import get_console_logger

logger = get_console_logger()


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
        predictions: Dict,
        *args,
        **kwargs,
    ) -> dict:
        if self.verbose:
            logger.log(f"Computing recall@{self.k}")

        # metrics to return
        metrics = {}

        stage = trainer.state.stage
        if stage not in {RunningStage.VALIDATING, RunningStage.TESTING}:
            raise ValueError(
                f"Stage {stage} not supported, only `validate` and `test` are supported."
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
        predictions: Dict,
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing recall@{self.k}")

        # metrics to return
        metrics = {}

        stage = trainer.state.stage
        if stage not in {RunningStage.VALIDATING, RunningStage.TESTING}:
            raise ValueError(
                f"Stage {stage} not supported, only `validate` and `test` are supported."
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
