import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Set

import hydra
import lightning as pl
import torch
from lightning.pytorch.trainer.states import RunningStage
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

import goldenretriever.common.dist_utils as dist
from goldenretriever.callbacks.base import (
    NLPTemplateCallback,
    PredictionCallback,
)
from goldenretriever.common.log import get_logger
from goldenretriever.data.base.datasets import BaseDataset
from goldenretriever.data.datasets import GoldenRetrieverStreamingDataset
from goldenretriever.indexers.base import BaseDocumentIndex
from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule
from goldenretriever.pytorch_modules.model import GoldenRetriever
from streaming.base.util import clean_stale_shared_memory

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")


class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: int | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        document_index: BaseDocumentIndex | DictConfig | None = None,
        precision: str | int = 32,
        force_reindex: bool = True,
        retriever_dir: Optional[Path] = None,
        stages: Set[str | RunningStage] | None = None,
        other_callbacks: List[DictConfig] | List["NLPTemplateCallback"] | None = None,
        dataset: DictConfig | BaseDataset | None = None,
        dataloader: DataLoader | None = None,
        # index: Optional[DictConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(batch_size, stages, other_callbacks, dataset, dataloader)
        self.k = k
        self.num_workers = num_workers
        self.document_index = document_index
        self.precision = precision
        self.force_reindex = force_reindex
        self.retriever_dir = retriever_dir

        if self.document_index is not None:
            if isinstance(self.document_index, str):
                self.document_index = BaseDocumentIndex.from_pretrained(
                    document_index  # , device=index_device, precision=index_precision, **kwargs
                )
            elif isinstance(self.document_index, DictConfig):
                self.document_index = hydra.utils.instantiate(document_index)
            elif not isinstance(self.document_index, BaseDocumentIndex):
                raise ValueError(
                    f"document_index must be either a string, a DictConfig or a BaseDocumentIndex, got {type(document_index)}"
                )

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule | GoldenRetrieverPLModule,
        datasets: (
            DictConfig | BaseDataset | List[DictConfig] | List[BaseDataset] | None
        ) = None,
        dataloaders: DataLoader | List[DataLoader] | None = None,
        limit_batches: int | None = None,
        *args,
        **kwargs,
    ) -> dict:
        stage = trainer.state.stage
        if trainer.global_rank == 0:
            logger.info(f"Computing predictions for stage {stage.value}")
        if stage not in self.stages:
            raise ValueError(
                f"Stage `{stage}` not supported, only {self.stages} are supported"
            )

        datasets, dataloaders = self._get_datasets_and_dataloaders(
            trainer,
            datasets,
            dataloaders,
            dataloader_kwargs=dict(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
            ),
        )

        # set the model to eval mode
        pl_module.eval()
        # check if the model is distributed and move it to the correct device
        # get the retriever
        retriever: GoldenRetriever = pl_module.model

        # here we will store the samples with predictions for each dataloader
        dataloader_predictions = {}
        # compute the passage embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            current_dataset: GoldenRetrieverStreamingDataset = datasets[dataloader_idx]
            if trainer.global_rank == 0:
                logger.info(
                    f"Computing passage embeddings for dataset {current_dataset.name}"
                )

            # check if we need to reindex the passages and
            # also if we need to load the retriever from disk
            if (self.retriever_dir is not None and trainer.current_epoch == 0) or (
                self.retriever_dir is not None and stage == RunningStage.TESTING
            ):
                force_reindex = False
            else:
                force_reindex = self.force_reindex

            if (
                not force_reindex
                and self.retriever_dir is not None
                and stage == RunningStage.TESTING
            ):
                retriever = retriever.from_pretrained(self.retriever_dir)

            # you never know :)
            retriever.eval()

            # if self.document_index is not None:
            #     if len(self.document_index) == 0:
            #         for sample in tqdm(
            #             current_dataset,
            #             desc=f"Adding documents from {current_dataset.name}",
            #         ):
            #             [
            #                 self.document_index.documents.add_document(s)
            #                 for s in sample["positives"]
            #             ]
            #             [
            #                 self.document_index.documents.add_document(s)
            #                 for s in sample["negatives"]
            #             ]
            #             [
            #                 self.document_index.documents.add_document(s)
            #                 for s in sample["hard_negatives"]
            #             ]
            #         clean_stale_shared_memory()

            retriever.index(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                max_length=current_dataset.max_passage_length,
                precision=self.precision,
                compute_on_cpu=False,
                force_reindex=force_reindex,
                document_index=self.document_index,
            )

            # sync the processes
            trainer.strategy.barrier()

            # now compute the question embeddings and compute the top-k accuracy
            predictions = []
            start = time.time()
            logger.info("Computing predictions")
            batch_augmented = 0

            progress_bar = tqdm(
                dataloader,
                initial=0,
                total=limit_batches or len(dataloader),
                desc=f"Computing predictions for dataset {current_dataset.name} on rank {trainer.global_rank}",
            )
            # for batch in tqdm(
            #     dataloader,
            #     desc=f"Computing predictions for dataset {current_dataset.name} on rank {trainer.global_rank}",
            # ):
            for batch in progress_bar:
                batch = batch.to(pl_module.device)
                # get the top-k indices
                retriever_output = retriever.retrieve(
                    **batch.questions,
                    k=self.k,
                    precision=self.precision,
                    document_index=self.document_index,
                )
                # compute recall at k
                for batch_idx, retrieved_samples in enumerate(retriever_output):
                    # get the positive passages
                    gold_passages = batch["positives"][batch_idx]
                    # get the index of the gold passages in the retrieved passages
                    gold_passage_indices = []
                    for passage in gold_passages:
                        try:
                            gold_passage_indices.append(
                                retriever.get_index_from_passage(passage)
                            )
                        except ValueError:
                            if trainer.global_rank == 0:
                                logger.warning(
                                    f"Passage `{passage}` not found in the index. "
                                    "We will skip it, but the results might not reflect the "
                                    "actual performance."
                                )
                            pass
                    retrieved_indices = [r.document.id for r in retrieved_samples if r]
                    retrieved_passages = [
                        retriever.get_passage_from_index(i) for i in retrieved_indices
                    ]
                    retrieved_scores = [r.score for r in retrieved_samples]
                    # correct predictions are the passages that are in the top-k and are gold
                    correct_indices = set(gold_passage_indices) & set(retrieved_indices)
                    # wrong predictions are the passages that are in the top-k and are not gold
                    wrong_indices = set(retrieved_indices) - set(gold_passage_indices)
                    # add the predictions to the list
                    prediction_output = dict(
                        sample_idx=batch.sample_idx[batch_idx],
                        gold=gold_passages,
                        predictions=retrieved_passages,
                        scores=retrieved_scores,
                        correct=[
                            retriever.get_passage_from_index(i) for i in correct_indices
                        ],
                        wrong=[
                            retriever.get_passage_from_index(i) for i in wrong_indices
                        ],
                    )
                    predictions.append(prediction_output)
                progress_bar.update(1)
                batch_augmented += 1
                if limit_batches is not None and batch_augmented >= limit_batches:
                    logger.info(
                        f"Augmented next iteration batches ({batch_augmented}). "
                        "Stopping the hard negative mining."
                    )
                    break
            end = time.time()
            logger.info(f"Time to retrieve: {str(end - start)}")
            trainer.strategy.barrier()
            # gather the predictions from all the processes
            predictions = dist.all_gather_object(predictions)
            # # predictions is now a list of lists of predictions, we need to merge them
            predictions = [p for sublist in predictions for p in sublist]
            dataloader_predictions[dataloader_idx] = predictions

        # return the predictions
        return dataloader_predictions
