# from pytorch_lightning import Callback, LightningModule
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging import get_console_logger

logger = get_console_logger()


class TopKEvaluationCallback(pl.Callback):
    def __init__(
        self, k: int = 100, report_intervals: Optional[int] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.report_intervals = report_intervals

    @torch.no_grad()
    def __call__(
        self,
        dataloaders: List[DataLoader],
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing top-k ({self.k}) accuracy")

        # retrieve the question and context encoders
        question_encoder = pl_module.model.question_encoder
        question_encoder.eval()
        context_encoder = pl_module.model.context_encoder
        context_encoder.eval()

        # metrics to return
        metrics = {}

        # compute the context embeddings index for each dataloader
        for i, dataloader in enumerate(dataloaders):
            # compute the context embeddings
            context_embeddings = []
            context_index = {}
            for batch in tqdm(dataloader, desc="Computing context embeddings"):
                batch = move_data_to_device(batch, pl_module.device)
                context_embeddings.append(context_encoder(**batch["contexts"]))
                for context_ids in batch["contexts"]["input_ids"]:
                    context_index[" ".join(map(str, context_ids.tolist()))] = len(context_index)
            context_embeddings = torch.cat(context_embeddings, dim=0)

            # now compute the question embeddings and compute the top-k accuracy
            recall_at_k_scores = []
            for batch in tqdm(dataloader, desc="Computing question scores"):
                batch = move_data_to_device(batch, pl_module.device)
                question_embeddings = question_encoder(**batch["questions"])
                # compute the similarity between the question and all the context embeddings
                similarity = torch.matmul(question_embeddings, context_embeddings.T)
                # get the top-k indices
                top_ks = torch.topk(similarity, k=self.k, dim=1).indices
                # compute recall at k
                recall_at_k = []
                for i, top_k in enumerate(top_ks):
                    labels = batch["labels"][i]
                    # get the positive context ids
                    positive_context_ids = [
                        context_ids
                        for context_ids, label in zip(
                            batch["contexts"]["input_ids"][i], labels
                        )
                        if label == 1
                    ]
                    # get the positive context indices
                    positive_context_indices = [
                        context_index[" ".join(map(str, context_ids.tolist()))]
                        for context_ids in positive_context_ids
                    ]
                    # compute the recall at k
                    recall_at_k.append(
                        len(set(top_k.tolist()) & set(positive_context_indices))
                        / len(positive_context_indices)
                    )
                recall_at_k = sum(recall_at_k) / len(recall_at_k)
                recall_at_k_scores.append(recall_at_k)

            mean_recall_at_k = sum(recall_at_k_scores) / len(recall_at_k_scores)
            # compute the mean recall at k
            metrics[f"recall_at_{self.k}_{i}"] = mean_recall_at_k

        return metrics

    def on_validation_epoch_end(self, trainer, pl_module):
        # transfer the dataloaders to the device
        metrics = self(trainer.val_dataloaders, pl_module)
        metrics = {f"val_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = self(trainer.test_dataloaders, pl_module)
        metrics = {f"test_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
