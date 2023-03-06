from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Optional, Set, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from utils.model_inputs import ModelInputs

from utils.logging import get_console_logger

from datasets import Dataset

# from faiss.indexer import FaissIndexer

logger = get_console_logger()


class NegativeAugmentationCallback(pl.Callback):
    def __init__(
        self,
        metric_to_monitor: str = "val_loss",
        threshold: float = 0.8,
        max_negatives: int = 3,
        batch_size: int = 32,
    ):
        self.metric_to_monitor = metric_to_monitor
        self.threshold = threshold
        self.max_negatives = max_negatives
        self.batch_size = batch_size

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.metric_to_monitor not in trainer.logged_metrics:
            raise ValueError(
                f"Metric {self.metric_to_monitor} not found in trainer.logged_metrics"
            )
        if trainer.logged_metrics[self.metric_to_monitor] < self.threshold:
            return

        logger.log(
            f"Metric {self.metric_to_monitor} is above threshold {self.threshold}. Augmenting the dataset."
        )
        # get the dataloaders and datasets
        dataloaders = trainer.val_dataloaders
        datasets = trainer.datamodule.val_datasets

        train_dataset = trainer.datamodule.train_dataset

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
                    collate_fn=lambda x: ModelInputs(
                        {
                            # this is a hack to normalize the batch structure
                            "contexts": trainer.datamodule.tokenizer(
                                x, padding=True, return_tensors="pt"
                            )
                        }
                    ),
                )
            else:
                context_dataloader = dataloader

            context_embeddings, context_index = self.compute_context_embeddings(
                pl_module.model.context_encoder,
                context_dataloader,
                pl_module.device,
            )

            # now compute the question embeddings and compute the top-k accuracy
            logger.log(
                f"Augmenting dataloader {dataloader_idx} with the top {self.max_negatives} negatives"
            )

            # create a defaultdict of defaultdicts to store the augmented contexts
            augmented_negative_contexts = defaultdict(lambda: defaultdict(list))

            for batch in dataloader:
                batch = batch.to(pl_module.device)
                model_outputs = pl_module.model(
                    batch.questions, contexts_encodings=context_embeddings
                )
                similarity = model_outputs["logits"]
                # get the top-k indices
                top_ks = torch.topk(similarity, k=similarity.shape[-1], dim=1).indices
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
                    # get the ids of the max_negatives wrong contexts with highest similarity
                    # wrong_context_ids = set(top_k_context_ids) - set(positive_context_ids)
                    wrong_context_ids = [
                        context_id
                        for context_id in top_k_context_ids
                        if context_id not in positive_context_ids
                    ][: self.max_negatives]

                    # get at most max_negatives wrong contexts
                    # wrong_context_ids = list(wrong_context_ids)[: self.max_negatives]
                    # convert the context ids to a list of ints
                    wrong_context_ids = [
                        [int(c) for c in context_id.split(" ")]
                        for context_id in wrong_context_ids
                    ]
                    # add the wrong contexts to the dataset sample
                    sample_idx_in_dataset = batch["ids"][sample_idx]
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
            dataset_dict = train_dataset.data.to_dict()
            dataset_dict["augmented_contexts"] = []
            # add the augmented contexts to the dataset
            for sample_idx in dataset_dict["id"]:
                sample_idx = int(sample_idx)
                if sample_idx in augmented_negative_contexts:
                    dataset_dict["augmented_contexts"].append(
                        augmented_negative_contexts[sample_idx]
                    )
            # create a new dataset
            train_dataset.data = Dataset.from_dict(dataset_dict)

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
