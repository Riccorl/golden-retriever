# from pytorch_lightning import Callback, LightningModule
import os
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm

# from faiss.indexer import FaissIndexer

from utils.logging import get_console_logger

logger = get_console_logger()


class TopKEvaluationCallback(pl.Callback):
    def __init__(
        self,
        k: int = 100,
        report_intervals: Optional[int] = None,
        contexts: Union[List[str], os.PathLike] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.report_intervals = report_intervals
        self.contexts = contexts

    @torch.no_grad()
    def __call__(
        self,
        dataloaders: List[DataLoader],
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing recall@{self.k}")

        # retrieve the question and context encoders
        question_encoder = pl_module.model.question_encoder
        question_encoder.eval()
        context_encoder = pl_module.model.context_encoder
        context_encoder.eval()

        # metrics to return
        metrics = {}

        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            # compute the context embeddings
            context_embeddings = []
            context_index = {}
            contexts = set()
            i = 0
            for batch in dataloader:
                batch = move_data_to_device(batch, pl_module.device)
                emb = context_encoder(**batch["contexts"])
                # context_embeddings.append()
                for context_ids, e in zip(batch["contexts"]["input_ids"], emb):
                    cleaned_context = " ".join(
                        map(str, [c for c in context_ids.tolist() if c != 0])
                    )
                    if cleaned_context not in contexts:
                        contexts.add(cleaned_context)
                        context_embeddings.append(e)
                        context_index[i] = cleaned_context
                        i += 1
            context_embeddings = torch.stack(context_embeddings, dim=0)
            # faiss_indexer = FaissIndexer(context_embeddings, normalize=True)

            # now compute the question embeddings and compute the top-k accuracy
            # recall_at_k_scores = []
            hits, total = 0, 0
            # for batch in tqdm(dataloader, desc="Computing question scores"):
            for batch in dataloader:
                batch = move_data_to_device(batch, pl_module.device)
                question_embeddings = question_encoder(**batch["questions"])
                # compute the similarity between the question and all the context embeddings
                similarity = torch.matmul(question_embeddings, context_embeddings.T)
                # get the top-k indices
                top_ks = torch.topk(similarity, k=self.k, dim=1).indices
                # top_ks = faiss_indexer.search(question_encoder(**batch["questions"]), k=self.k)
                # compute recall at k
                recall_at_k = []
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

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = self(trainer.val_dataloaders, pl_module)
        metrics = {f"val_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = self(trainer.test_dataloaders, pl_module)
        metrics = {f"test_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
