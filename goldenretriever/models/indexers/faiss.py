import contextlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy
import torch
import tqdm
from omegaconf import OmegaConf
from rich.pretty import pprint
from torch.utils.data import DataLoader

from goldenretriever.common.log import get_console_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.common.utils import is_package_available
from goldenretriever.data.base.datasets import BaseDataset
from goldenretriever.data.labels import Labels
from goldenretriever.models import PRECISION_MAP
from goldenretriever.models.indexers.base import BaseIndexer
from goldenretriever.models.model import GoldenRetriever, RetrievedSample

if is_package_available("faiss"):
    import faiss
    import faiss.contrib.torch_utils

logger = get_console_logger()


@dataclass
class FaissOutput:
    indices: Union[torch.Tensor, numpy.ndarray]
    distances: Union[torch.Tensor, numpy.ndarray]


class FaissIndexer(BaseIndexer):
    DOCUMENTS_FILE_NAME = "documents.json"
    EMBEDDINGS_FILE_NAME = "embeddings.bin"

    def __init__(
        self,
        documents: Union[List[str], Labels],
        embeddings: Optional[Union[torch.Tensor, numpy.ndarray]] = None,
        index=None,
        index_type: str = "Flat",
        metric: int = faiss.METRIC_INNER_PRODUCT,
        normalize: bool = False,
        device: str = "cpu",
        *args,
        **kwargs,
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

        # device to store the embeddings
        self.device = device

        # params
        self.index_type = index_type
        self.metric = metric
        self.normalize = normalize

        if index is not None:
            self.index = index
            if self.device == "cuda":
                # use a single GPU
                faiss_resource = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(faiss_resource, 0, self.index)
        else:
            # build the faiss index
            self._build_faiss_index(
                embeddings=embeddings,
                index_type=index_type,
                normalize=normalize,
                metric=metric,
            )

        # embeddings of the documents
        self.embeddings = index

    def _build_faiss_index(
        self,
        embeddings: Optional[Union[torch.Tensor, numpy.ndarray]],
        index_type: str,
        normalize: bool,
        metric: int,
    ):
        # build the faiss index
        self.normalize = (
            normalize
            and metric == faiss.METRIC_INNER_PRODUCT
            and not isinstance(embeddings, torch.Tensor)
        )
        if self.normalize:
            index_type = f"L2norm,{index_type}"
        faiss_vector_size = embeddings.shape[1]
        if self.device == "cpu":
            index_type = index_type.replace("x,", "x_HNSW32,")
        index_type = index_type.replace(
            "x", str(math.ceil(math.sqrt(faiss_vector_size)) * 4)
        )
        self.index = faiss.index_factory(faiss_vector_size, index_type, metric)

        # convert to GPU
        if self.device == "cuda":
            # use a single GPU
            faiss_resource = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(faiss_resource, 0, self.index)
        else:
            # move to CPU if embeddings is a torch.Tensor
            embeddings = (
                embeddings.cpu() if isinstance(embeddings, torch.Tensor) else embeddings
            )

        # convert to float32 if embeddings is a torch.Tensor and is float16
        if isinstance(embeddings, torch.Tensor) and embeddings.dtype == torch.float16:
            embeddings = embeddings.float()

        self.index.add(embeddings)

        # save parameters for saving/loading
        self.index_type = index_type
        self.metric = metric

        return self.index

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
    ) -> torch.Tensor:
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
        # if self.device == "cpu":
        passage_embeddings = [c.detach().cpu() for c in passage_embeddings]
        # stack it
        passage_embeddings: torch.Tensor = torch.stack(passage_embeddings, dim=0)
        # convert to float32 for faiss
        passage_embeddings.to(PRECISION_MAP["float32"])
        self.embeddings = passage_embeddings

        # free up memory from the unused variable
        del passage_embeddings

        # index the embeddings
        self._build_faiss_index(
            embeddings=self.embeddings,
            index_type=self.index_type,
            normalize=self.normalize,
            metric=self.metric,
        )
        return self.embeddings

    @torch.no_grad()
    @torch.inference_mode()
    def search(self, query: torch.Tensor, k: int = 1) -> List[RetrievedSample]:
        k = min(k, self.index.ntotal)

        if self.normalize:
            faiss.normalize_L2(query)
        if isinstance(query, torch.Tensor) and not self.use_gpu:
            query = query.detach().cpu()
        # Retrieve the indices of the top k passage embeddings
        retriever_out = self.index.search(query, k)

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
        """
        Save the indexer to the disk.

        Args:
            saving_dir (:obj:`Union[str, os.PathLike]`):
                The directory where the indexer will be saved.
        """
        saving_dir = Path(saving_dir)
        # save the passage embeddings
        embedding_path = saving_dir / self.EMBEDDING_FILE_NAME
        logger.info(f"Saving passage embeddings to {embedding_path}")
        faiss.write_index(self.index, str(embedding_path))
        # save the passage index
        documents_path = saving_dir / self.DOCUMENTS_FILE_NAME
        logger.info(f"Saving passage index to {documents_path}")
        self.documents.save(documents_path)

    @classmethod
    def load(
        cls,
        loading_dir: Union[str, os.PathLike],
        device: str = "cpu",
        document_file_name: Optional[str] = None,
        embedding_file_name: Optional[str] = None,
        **kwargs,
    ) -> "FaissIndexer":
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
            raise ValueError(f"Index file `{embedding_path}` does not exist.")
        logger.info(f"Loading index from {embedding_path}")
        index = faiss.read_index(str(embedding_path))

        return cls(documents=documents, index=index, device=device, **kwargs)

    def get_embeddings_from_index(
        self, index: int
    ) -> Union[torch.Tensor, numpy.ndarray]:
        """
        Get the document vector from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.index is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        if index >= self.index.ntotal:
            raise ValueError(
                f"The index {index} is out of bounds. The maximum index is {self.index.ntotal}."
            )
        return self.index.reconstruct(index)
