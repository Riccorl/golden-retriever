import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy
import torch
from torch.utils.data import DataLoader
import tqdm

from goldenretriever.common.log import get_console_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.base.datasets import BaseDataset
from goldenretriever.data.labels import Labels
from goldenretriever.models import PRECISION_MAP
from goldenretriever.models.model import GoldenRetriever, RetrievedSample


logger = get_console_logger()


@dataclass
class IndexerOutput:
    indices: Union[torch.Tensor, numpy.ndarray]
    distances: Union[torch.Tensor, numpy.ndarray]


class InMemoryIndexer:
    DOCUMENTS_FILE_NAME = "documents.json"
    EMBEDDINGS_FILE_NAME = "embeddings.pt"

    def __init__(
        self,
        documents: Union[List[str], Labels],
        embeddings: Optional[Union[torch.Tensor, numpy.ndarray]] = None,
        device: str = "cpu",
    ) -> None:
        if embeddings is not None and documents is not None:
            logger.log("Both documents and embeddings are provided.")
            if len(documents) != embeddings.shape[0]:
                raise ValueError(
                    "The number of documents and embeddings must be the same."
                )

        # documents to be used for indexing
        if isinstance(documents, Labels):
            self.documents = documents
        else:
            self.documents = Labels()
            self.documents.add_labels(documents)

        # embeddings of the documents
        self.embeddings = embeddings

        # device to store the embeddings
        self.device = device

    @torch.no_grad()
    @torch.inference_mode()
    def index(
        self,
        retriever: GoldenRetriever,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        encoder_precision: Optional[Union[str, int]] = None,
        precision: Optional[Union[str, int]] = None,
        compute_on_cpu: bool = False,
        force_reindex: bool = False,
    ) -> "InMemoryIndexer":
        """
        Index the documents using the encoder.

        Args:
            encoder (:obj:`torch.nn.Module`):
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
            precision (:obj:`Union[str, int]`, `optional`, defaults to None):
                The precision to be used for the embeddings.
            compute_on_cpu (:obj:`bool`, `optional`, defaults to False):
                Whether to compute the embeddings on CPU.
            force_reindex (:obj:`bool`, `optional`, defaults to False):
                Whether to force reindexing.
            update_existing (:obj:`bool`, `optional`, defaults to False):
                Whether to update the existing embeddings.
            duplicate_strategy (:obj:`str`, `optional`, defaults to "overwrite"):
                The strategy to be used for duplicate embeddings. Can be one of "overwrite" or "ignore".

        Returns:
            :obj:`InMemoryIndexer`: The indexer object.
        """

        if self.embeddings is not None and not force_reindex:
            logger.log(
                "Embeddings are already present and `force_reindex` is `False`. Skipping indexing."
            )
            return self

        # release the memory
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
        dataloader = DataLoader(
            BaseDataset(name="passage", data=self.documents.get_passages()),
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
            contextlib.nullpassage()
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
                passage_outs = encoder(**batch)
                # Append the passage embeddings to the list
                if self.device == "cpu":
                    passage_embeddings.extend([c.detach().cpu() for c in passage_outs])
                else:
                    passage_embeddings.extend([c for c in passage_outs])

        # move the passage embeddings to the CPU if not already done
        # the move to cpu and then to gpu is needed to avoid OOM when using mixed precision
        if self.device == "cpu":
            passage_embeddings = [c.detach().cpu() for c in passage_embeddings]
        # stack it
        passage_embeddings: torch.Tensor = torch.stack(passage_embeddings, dim=0)
        # move the passage embeddings to the gpu if needed
        if not self.device == "cpu":
            if precision:
                passage_embeddings = passage_embeddings.to(PRECISION_MAP[precision])
            passage_embeddings = passage_embeddings.to(self.device)
        self.embeddings = passage_embeddings

        # free up memory from the unused variable
        del passage_embeddings

        return self.embeddings

    @torch.no_grad()
    @torch.inference_mode()
    def search(self, query: torch.Tensor, k: int = 1) -> List[RetrievedSample]:
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

    def save(self, saving_dir: Union[str, os.PathLike]):
        saving_dir = Path(saving_dir)

        # save the passage embeddings
        embedding_path = saving_dir / self.EMBEDDING_FILE_NAME
        logger.info(f"Saving passage embeddings to {embedding_path}")
        torch.save(self._passage_embeddings, embedding_path)
        # save the passage index
        documents_path = saving_dir / self.DOCUMENTS_FILE_NAME
        logger.info(f"Saving passage index to {documents_path}")
        self.documents.save(documents_path)

    @classmethod
    def load(
        cls,
        loading_dir: Union[str, os.PathLike],
        precision: Optional[str] = None,
        device: str = "cpu",
        document_file_name: Optional[str] = None,
        embedding_file_name: Optional[str] = None,
        **kwargs,
    ) -> "InMemoryIndexer":
        loading_dir = Path(loading_dir)

        document_file_name = document_file_name or cls.DOCUMENTS_FILE_NAME
        embedding_file_name = embedding_file_name or cls.EMBEDDING_FILE_NAME

        # load the documents
        documents_path = loading_dir / document_file_name

        if not documents_path.exists():
            raise ValueError(f"Document file `{documents_path}` does not exist.")
        logger.info(f"Loading documents from {documents_path}")
        documents = Labels.from_file(documents_path)

        # load the passage embeddings
        embedding_path = loading_dir / embedding_file_name
        # run some checks
        if not embedding_path.exists():
            raise ValueError(f"Embedding file `{embedding_path}` does not exist.")
        logger.info(f"Loading embeddings from {embedding_path}")

        embeddings = torch.load(embedding_path, map_location="cpu")
        if precision is not None:
            if embedding_path and embeddings.dtype != PRECISION_MAP[precision]:
                logger.info(
                    f"Index vectors are of type {embeddings.dtype}. "
                    f"Converting to {PRECISION_MAP[precision]}."
                )
                embeddings = embeddings.to(PRECISION_MAP[precision])

        if device == "cuda":
            embeddings = embeddings.to(device)

        return cls(documents=documents, embeddings=embeddings, device=device, **kwargs)
