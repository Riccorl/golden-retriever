import contextlib
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from goldenretriever.common.log import get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.base.datasets import BaseDataset
from goldenretriever.data.labels import Labels
from goldenretriever.retriever import PRECISION_MAP
from goldenretriever.retriever.indexers.base import BaseDocumentIndex
from retriever import RetrievedSample

logger = get_logger(__name__, level=logging.INFO)


class InMemoryDocumentIndex(BaseDocumentIndex):
    DOCUMENTS_FILE_NAME = "documents.json"
    EMBEDDINGS_FILE_NAME = "embeddings.pt"

    def __init__(
        self,
        documents: Union[str, List[str], Labels, os.PathLike, List[os.PathLike]],
        embeddings: Optional[torch.Tensor] = None,
        device: str = "cpu",
        precision: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        An in-memory indexer.

        Args:
            documents (:obj:`Union[List[str], PassageManager]`):
                The documents to be indexed.
            embeddings (:obj:`Optional[torch.Tensor]`, `optional`, defaults to :obj:`None`):
                The embeddings of the documents.
            device (:obj:`str`, `optional`, defaults to "cpu"):
                The device to be used for storing the embeddings.
        """

        super().__init__(documents, embeddings)

        if embeddings is not None and documents is not None:
            logger.info("Both documents and embeddings are provided.")
            if documents.get_label_size() != embeddings.shape[0]:
                raise ValueError(
                    "The number of documents and embeddings must be the same."
                )

        # embeddings of the documents
        self.embeddings = embeddings
        # does this do anything?
        del embeddings
        # convert the embeddings to the desired precision
        if precision is not None:
            if (
                self.embeddings is not None
                and self.embeddings.dtype != PRECISION_MAP[precision]
            ):
                logger.info(
                    f"Index vectors are of type {self.embeddings.dtype}. "
                    f"Converting to {PRECISION_MAP[precision]}."
                )
                self.embeddings = self.embeddings.to(PRECISION_MAP[precision])
        # move the embeddings to the desired device
        if self.embeddings is not None and not self.embeddings.device == device:
            self.embeddings = self.embeddings.to(device)

        # device to store the embeddings
        self.device = device
        # precision to be used for the embeddings
        self.precision = precision

    @torch.no_grad()
    @torch.inference_mode()
    def index(
        self,
        retriever,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        encoder_precision: Optional[Union[str, int]] = None,
        compute_on_cpu: bool = False,
        force_reindex: bool = False,
    ) -> "InMemoryDocumentIndex":
        """
        Index the documents using the encoder.

        Args:
            retriever (:obj:`torch.nn.Module`):
                The encoder to be used for indexing.
            batch_size (:obj:`int`, `optional`, defaults to 32):
                The batch size to be used for indexing.
            num_workers (:obj:`int`, `optional`, defaults to 4):
                The number of workers to be used for indexing.
            max_length (:obj:`int`, `optional`, defaults to None):
                The maximum length of the input to the encoder.
            collate_fn (:obj:`Callable`, `optional`, defaults to None):
                The collate function to be used for batching.
            encoder_precision (:obj:`Union[str, int]`, `optional`, defaults to None):
                The precision to be used for the encoder.
            compute_on_cpu (:obj:`bool`, `optional`, defaults to False):
                Whether to compute the embeddings on CPU.
            force_reindex (:obj:`bool`, `optional`, defaults to False):
                Whether to force reindexing.

        Returns:
            :obj:`InMemoryIndexer`: The indexer object.
        """

        if self.embeddings is not None and not force_reindex:
            logger.info(
                "Embeddings are already present and `force_reindex` is `False`. Skipping indexing."
            )
            return self

        if collate_fn is None:
            tokenizer = retriever.passage_tokenizer
            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length or tokenizer.model_max_length,
                )
            )
        data = [k for k in self.documents.get_labels()]
        dataloader = DataLoader(
            BaseDataset(name="passage", data=data),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        encoder = retriever.passage_encoder

        # Create empty lists to store the passage embeddings and passage index
        passage_embeddings: List[torch.Tensor] = []

        encoder_device = "cpu" if compute_on_cpu else self.device

        # fucking autocast only wants pure strings like 'cpu' or 'cuda'
        # we need to convert the model device to that
        device_type_for_autocast = str(encoder_device).split(":")[0]
        # autocast doesn't work with CPU and stuff different from bfloat16
        autocast_pssg_mngr = (
            contextlib.nullcontext()
            if device_type_for_autocast == "cpu"
            else (
                torch.autocast(
                    device_type=device_type_for_autocast,
                    dtype=PRECISION_MAP[encoder_precision],
                )
            )
        )
        with autocast_pssg_mngr:
            # Iterate through each batch in the dataloader
            for batch in tqdm(dataloader, desc="Indexing"):
                # Move the batch to the device
                batch: ModelInputs = batch.to(encoder_device)
                # Compute the passage embeddings
                passage_outs = encoder(**batch).pooler_output
                # Append the passage embeddings to the list
                if self.device == "cpu":
                    passage_embeddings.extend([c.detach().cpu() for c in passage_outs])
                else:
                    passage_embeddings.extend([c for c in passage_outs])

        # move the passage embeddings to the CPU if not already done
        # the move to cpu and then to gpu is needed to avoid OOM when using mixed precision
        if not self.device == "cpu":  # this if is to avoid unnecessary moves
            passage_embeddings = [c.detach().cpu() for c in passage_embeddings]
        # stack it
        passage_embeddings: torch.Tensor = torch.stack(passage_embeddings, dim=0)
        # move the passage embeddings to the gpu if needed
        if not self.device == "cpu":
            passage_embeddings = passage_embeddings.to(PRECISION_MAP[self.precision])
            passage_embeddings = passage_embeddings.to(self.device)
        self.embeddings = passage_embeddings

        # free up memory from the unused variable
        del passage_embeddings

        return self

    @torch.no_grad()
    @torch.inference_mode()
    def search(self, query: torch.Tensor, k: int = 1) -> list[list[RetrievedSample]]:
        """
        Search the documents using the query.

        Args:
            query (:obj:`torch.Tensor`):
                The query to be used for searching.
            k (:obj:`int`, `optional`, defaults to 1):
                The number of documents to be retrieved.

        Returns:
            :obj:`List[RetrievedSample]`: The retrieved documents.
        """
        # fucking autocast only wants pure strings like 'cpu' or 'cuda'
        # we need to convert the model device to that
        device_type_for_autocast = str(self.device).split(":")[0]
        # autocast doesn't work with CPU and stuff different from bfloat16
        autocast_pssg_mngr = (
            contextlib.nullcontext()
            if device_type_for_autocast == "cpu"
            else (
                torch.autocast(
                    device_type=device_type_for_autocast,
                    dtype=self.embeddings.dtype,
                )
            )
        )
        with autocast_pssg_mngr:
            similarity = torch.matmul(query, self.embeddings.T)
            # Retrieve the indices of the top k passage embeddings
            retriever_out: Tuple = torch.topk(
                similarity, k=min(k, similarity.shape[-1]), dim=1
            )
        # get int values
        batch_top_k: List[List[int]] = retriever_out.indices.detach().cpu().tolist()
        # get float values
        batch_scores: List[List[float]] = retriever_out.values.detach().cpu().tolist()
        # Retrieve the passages corresponding to the indices
        batch_passages = [
            [self.documents.get_label_from_index(i) for i in indices]
            for indices in batch_top_k
        ]
        # build the output object
        batch_retrieved_samples = [
            [
                RetrievedSample(label=passage, index=index, score=score)
                for passage, index, score in zip(passages, indices, scores)
            ]
            for passages, indices, scores in zip(
                batch_passages, batch_top_k, batch_scores
            )
        ]
        return batch_retrieved_samples
