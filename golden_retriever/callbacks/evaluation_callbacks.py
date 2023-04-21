from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch

from sklearn.metrics import label_ranking_average_precision_score

from golden_retriever.callbacks.base import DEFAULT_STAGES, NLPTemplateCallback
from golden_retriever.common.logging import get_console_logger

logger = get_console_logger()


class TopKEvaluationCallback(NLPTemplateCallback):
    def __init__(
        self,
        k: int = 100,
        prefix: Optional[str] = None,
        verbose: bool = False,
        prog_bar: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.prefix = prefix
        self.verbose = verbose
        self.prog_bar = prog_bar

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
        if stage not in DEFAULT_STAGES:
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
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=self.prog_bar
        )

        if self.verbose:
            logger.log(
                f"Recall@{self.k} on {stage.value}: {metrics[f'{stage.value}_recall@{self.k}']}"
            )

        return metrics


class LRAPEvaluationCallback(NLPTemplateCallback):
    def __init__(
        self,
        k: int = 100,
        prefix: Optional[str] = None,
        verbose: bool = False,
        prog_bar: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.prefix = prefix
        self.verbose = verbose
        self.prog_bar = prog_bar

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
        if stage not in DEFAULT_STAGES:
            raise ValueError(
                f"Stage {stage} not supported, only `validate` and `test` are supported."
            )

        for dataloader_idx, samples in predictions.items():
            scores = [sample["scores"][: self.k] for sample in samples]
            golds = [sample["gold"] for sample in samples]

            # compute the mean recall at k
            lrap = label_ranking_average_precision_score(golds, scores)
            metrics[f"lrap@{self.k}_{dataloader_idx}"] = lrap
        metrics[f"lrap@{self.k}"] = sum(metrics.values()) / len(metrics)

        if self.prefix is not None:
            metrics = {f"{self.prefix}_{k}": v for k, v in metrics.items()}
        else:
            metrics = {f"{stage.value}_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=self.prog_bar
        )

        if self.verbose:
            logger.log(
                f"Recall@{self.k} on {stage.value}: {metrics[f'{stage.value}_recall@{self.k}']}"
            )

        return metrics


class AvgRankingEvaluationCallback(NLPTemplateCallback):
    def __init__(
        self,
        k: int,
        prefix: Optional[str] = None,
        verbose: bool = True,
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
        if stage not in DEFAULT_STAGES:
            raise ValueError(
                f"Stage {stage} not supported, only `validate` and `test` are supported."
            )

        for dataloader_idx, samples in predictions.items():
            rankings = []
            for sample in samples:
                window_candidates = sample["predictions"][: self.k]
                window_labels = sample["gold"]
                for wl in window_labels:
                    if wl in window_candidates:
                        rankings.append(window_candidates.index(wl) + 1)

            avg_ranking = sum(rankings) / len(rankings) if len(rankings) > 0 else 0
            metrics[f"avg_ranking@{self.k}_{dataloader_idx}"] = avg_ranking
        if len(metrics) == 0:
            metrics[f"avg_ranking@{self.k}"] = 0
        else:
            metrics[f"avg_ranking@{self.k}"] = sum(metrics.values()) / len(metrics)

        prefix = self.prefix or stage.value
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)

        if self.verbose:
            logger.log(
                f"AVG Ranking@{self.k} on {prefix}: {metrics[f'{prefix}_avg_ranking@{self.k}']}"
            )

        return metrics


class NYTTopKEvaluationCallback(TopKEvaluationCallback):
    def __init__(
        self,
        label_mapping: Dict[str, List[str]],
        k: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(k, *args, **kwargs)
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
        if stage not in DEFAULT_STAGES:
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
