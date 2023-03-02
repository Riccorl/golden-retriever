# from pytorch_lightning import Callback, LightningModule
from typing import Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from utils.logging import get_console_logger

# from faiss.indexer import FaissIndexer

logger = get_console_logger()


class TopKEvaluationCallback(pl.Callback):
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
        stage: str,
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing recall@{self.k}")

        # metrics to return
        metrics = {}

        if stage not in ["validation", "test"]:
            raise ValueError(
                f"Stage {stage} not supported, only `validation` and `test` are supported."
            )

        dataloaders = (
            trainer.val_dataloaders
            if stage == "validation"
            else trainer.test_dataloaders
        )
        datasets = (
            trainer.datamodule.val_datasets
            if stage == "validation"
            else trainer.datamodule.test_datasets
        )

        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            logger.log(f"Computing context embeddings for dataloader {dataloader_idx}")
            if datasets[dataloader_idx].contexts is not None:
                context_dataloader = DataLoader(
                    datasets[dataloader_idx].contexts,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=lambda x: {
                        # this is a hack to normalize the batch structure
                        "contexts": trainer.datamodule.tokenizer(
                            x, padding=True, return_tensors="pt"
                        )
                    },
                )
            else:
                context_dataloader = dataloader

            context_embeddings, context_index = self.compute_context_embeddings(
                pl_module.model.context_encoder, context_dataloader, pl_module.device
            )

            # now compute the question embeddings and compute the top-k accuracy
            logger.log(f"Computing recall@{self.k} for dataloader {dataloader_idx}")
            hits, total = 0, 0
            for batch in dataloader:
                batch = batch.to(pl_module.device)
                similarity = pl_module.model(
                    batch.questions, contexts_encodings=context_embeddings
                )
                # get the top-k indices
                top_ks = torch.topk(
                    similarity, k=min(self.k, similarity.shape[-1]), dim=1
                ).indices
                # compute recall at k
                for sample_idx, top_k in enumerate(top_ks):
                    labels = batch["labels_for_metrics"][sample_idx]
                    # get the positive context ids
                    positive_context_ids = [
                        " ".join(map(str, [c for c in context_ids.tolist() if c != 0]))
                        for context_ids, label in zip(
                            batch["contexts"]["input_ids"], labels
                        )
                        if label == 1
                    ]
                    # get the top_k context ids
                    top_k_context_ids = [
                        context_index[context_idx] for context_idx in top_k.tolist()
                    ]
                    # compute the recall at k
                    hits += len(set(top_k_context_ids) & set(positive_context_ids))
                    total += len(set(positive_context_ids))

            # compute the mean recall at k
            recall_at_k = hits / total
            metrics[f"recall@{self.k}_{dataloader_idx}"] = recall_at_k
        metrics[f"recall@{self.k}"] = sum(metrics.values()) / len(metrics)
        return metrics

    @staticmethod
    @torch.no_grad()
    def compute_context_embeddings(
        context_encoder: torch.nn.Module,
        dataloader: DataLoader,
        device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, Dict[int, str]]:
        context_embeddings = []
        context_index = {}
        already_seen = set()
        i = 0
        for batch in dataloader:
            batch = batch.to(device)
            context_outs = context_encoder(**batch.contexts)
            for context_ids, e in zip(batch.contexts.input_ids, context_outs):
                cleaned_context = " ".join(
                    map(str, [c for c in context_ids.tolist() if c != 0])
                )
                if cleaned_context not in already_seen:
                    already_seen.add(cleaned_context)
                    context_embeddings.append(e)
                    context_index[i] = cleaned_context
                    i += 1
        context_embeddings = torch.stack(context_embeddings, dim=0)
        return context_embeddings, context_index

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        metrics = self(trainer, pl_module, "validation")
        metrics = {f"val_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = self(trainer, pl_module, "test")
        metrics = {f"test_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
