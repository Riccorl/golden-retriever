import json
import os
import tempfile
from typing import Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import goldenretriever.common.dist_utils as dist
from goldenretriever.common.data_utils import preprocess_to_mds
from goldenretriever.common.log import get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.common.torch_utils import get_autocast_context
from goldenretriever.data.base.datasets import BaseDataset
from goldenretriever.data.datasets import TxtStreamingDataset
from goldenretriever.indexers.base import BaseDocumentIndex
from goldenretriever.indexers.document import Document, DocumentStore
from goldenretriever.pytorch_modules import PRECISION_MAP, RetrievedSample

logger = get_logger(__name__)


class InMemoryDocumentIndex(BaseDocumentIndex):
    DOCUMENTS_FILE_NAME = "documents.jsonl"
    EMBEDDINGS_FILE_NAME = "embeddings.pt"

    def __init__(
        self,
        documents: (
            str | List[str] | os.PathLike | List[os.PathLike] | DocumentStore | None
        ) = None,
        embeddings: torch.Tensor | None = None,
        metadata_fields: List[str] | None = None,
        separator: str | None = None,
        name_or_path: str | os.PathLike | None = None,
        device: str = "cpu",
        precision: str | int | torch.dtype = 32,
        multi_gpu_indexing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        An in-memory indexer based on PyTorch.

        Args:
            documents (:obj:`Union[List[str]]`):
                The documents to be indexed.
            embeddings (:obj:`Optional[torch.Tensor]`, `optional`, defaults to :obj:`None`):
                The embeddings of the documents.
            device (:obj:`str`, `optional`, defaults to "cpu"):
                The device to be used for storing the embeddings.
        """

        super().__init__(
            documents, embeddings, metadata_fields, separator, name_or_path, device
        )

        if embeddings is not None and documents is not None:
            logger.info("Both documents and embeddings are provided.")
            if len(documents) != embeddings.shape[0]:
                raise ValueError(
                    "The number of documents and embeddings must be the same."
                )

        # # embeddings of the documents
        # self.embeddings = embeddings
        # does this do anything?
        del embeddings
        # convert the embeddings to the desired precision
        if precision is not None:
            if self.embeddings is not None and device == "cpu":
                if PRECISION_MAP[precision] == PRECISION_MAP[16]:
                    logger.info(
                        f"Precision `{precision}` is not supported on CPU. "
                        f"Using `{PRECISION_MAP[32]}` instead."
                    )
                precision = 32

            if (
                self.embeddings is not None
                and self.embeddings.dtype != PRECISION_MAP[precision]
            ):
                logger.info(
                    f"Index vectors are of type {self.embeddings.dtype}. "
                    f"Converting to {PRECISION_MAP[precision]}."
                )
                self.embeddings = self.embeddings.to(PRECISION_MAP[precision])
        else:
            # TODO: a bit redundant, fix this eventually
            if (
                device == "cpu"
                and self.embeddings is not None
                and self.embeddings.dtype != torch.float32
            ):
                logger.info(
                    f"Index vectors are of type {self.embeddings.dtype}. "
                    f"Converting to {PRECISION_MAP[32]}."
                )
                self.embeddings = self.embeddings.to(PRECISION_MAP[32])
        # move the embeddings to the desired device
        if self.embeddings is not None and not self.embeddings.device == device:
            self.embeddings = self.embeddings.to(device)

        # TODO: check interactions with the embeddings
        # self.mm = MatrixMultiplicationModule(embeddings=self.embeddings)
        # self.mm.eval()

        # precision to be used for the embeddings
        self.precision = precision

        self.multi_gpu_indexing = multi_gpu_indexing

    @torch.no_grad()
    @torch.inference_mode()
    def index(
        self,
        retriever,
        documents: Optional[List[Document]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int | None = None,
        collate_fn: Optional[Callable] = None,
        encoder_precision: Optional[Union[str, int]] = None,
        compute_on_cpu: bool = False,
        force_reindex: bool = False,
        dataloader: DataLoader | None = None,
        low_gpu_memory: bool = False,
    ) -> "InMemoryDocumentIndex":
        """
        Index the documents using the encoder.

        Args:
            retriever (:obj:`torch.nn.Module`):
                The encoder to be used for indexing.
            documents (:obj:`List[Document]`, `optional`, defaults to :obj:`None`):
                The documents to be indexed. If not provided, the documents provided at the initialization will be used.
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
            dataloader (:obj:`DataLoader`, `optional`, defaults to None):
                The dataloader to be used for indexing.
            low_gpu_memory (:obj:`bool`, `optional`, defaults to False):
                Whether if there is a low GPU memory environment.

        Returns:
            :obj:`InMemoryIndexer`: The indexer object.
        """

        if documents is None and self.documents is None:
            raise ValueError("Documents must be provided.")

        if self.embeddings is not None and not force_reindex and documents is None:
            logger.info(
                "Embeddings are already present and `force_reindex` is `False`. Skipping indexing."
            )
            return self

        if force_reindex:
            if documents is not None:
                self.documents.add_documents(documents)
            data = [k for k in self.get_passages()]

        else:
            if documents is not None:
                data = [k for k in self.get_passages(DocumentStore(documents))]
                # add the documents to the actual document store
                self.documents.add_documents(documents)
            else:
                if self.embeddings is None:
                    data = [k for k in self.get_passages()]

        if collate_fn is None:
            tokenizer = retriever.passage_tokenizer

            def collate_fn(batch):
                tokenized = tokenizer(
                    [x["text"] for x in batch],
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length or tokenizer.model_max_length,
                )
                tokenized.update({"id": [x["id"] for x in batch]})
                return ModelInputs(tokenized)

        # dataset_class = TxtStreamingDataset if self.multi_gpu_indexing else BaseDataset
        # if self.multi_gpu_indexing:
        #     # here we will process the passages as MDS files to be compatible with the streaming dataset
        #     # but we do it only on the rank 0, other ranks will receive the path to the processed data
        #     data_path = [None]  # this is needed to broadcast the data path
        #     if dist.get_rank() == 0:
        #         # save data in a temp file
        #         with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as f:
        #             for i, sample in enumerate(data):
        #                 f.write(json.dumps({"text": sample, "id": i}) + "\n")
        #             f.close()
        #             data_path = preprocess_to_mds(f.name)
        #         # delete the temporary file
        #         os.remove(f.name)
        #         # wrap the data path in a list to broadcast it
        #         data_path = [data_path]

        #     dist.broadcast_object_list(data_path, src=0)
        #     # extract the data path from the list
        #     data_path = data_path[0]
        #     dataset = TxtStreamingDataset(
        #         name="passage", local=data_path, batch_size=batch_size
        #     )
        # else:
        dataset = BaseDataset(
            name="passage", data=[{"text": x, "id": i} for i, x in enumerate(data)]
        )

        dataloader = DataLoader(
            # BaseDataset(name="passage", data=data),
            # TxtStreamingDataset(name="passage", local=data_path, batch_size=batch_size),
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            prefetch_factor=(
                max(1, 8 * batch_size // num_workers) if num_workers > 0 else None
            ),
        )

        encoder = retriever.passage_encoder
        encoder_device = "cpu" if compute_on_cpu else encoder.device

        passage_embeddings_dict: Dict[int, torch.Tensor] = {}

        # if self.multi_gpu_indexing or dist.get_rank() == 0:
        with get_autocast_context(encoder_device, encoder_precision):
            # Iterate through each batch in the dataloader
            for batch in tqdm(
                dataloader, desc=f"Indexing at rank {dist.get_rank()}"
            ):
                # Move the batch to the device
                passages_ids = batch.pop("id")
                batch: ModelInputs = batch.to(encoder_device)
                # Compute the passage embeddings
                passage_outs = encoder(**batch).pooler_output
                # Append the passage embeddings to the list
                if self.device == "cpu":
                    passage_embeddings_dict.update(
                        {
                            i: c.detach().cpu()
                            for i, c in zip(passages_ids, passage_outs)
                        }
                    )
                else:
                    # passage_embeddings.extend([c for c in passage_outs])
                    passage_embeddings_dict.update(
                        {i: c for i, c in zip(passages_ids, passage_outs)}
                    )

        # move the passage embeddings to the CPU if not already done
        # the move to cpu and then to gpu is needed to avoid OOM when using mixed precision
        # move them only if there is a low GPU memory environment
        if not self.device == "cpu" and low_gpu_memory:
            passage_embeddings_dict = {
                i: c.detach().cpu() for i, c in passage_embeddings_dict.items()
            }
        # synchronize the processes
        dist.barrier()
        # if dist.get_rank() == 0:
        #     if self.multi_gpu_indexing:
        #         logger.debug(f"All-gathering embeddings at rank {dist.get_rank()}")
        #         passage_embeddings_dict = dist.all_gather_object(
        #             passage_embeddings_dict
        #         )
        #         logger.debug(f"Received embeddings at rank {dist.get_rank()}")

        #         # merge the passage embeddings from all the devices
        #         passage_embeddings = {}
        #         for d in passage_embeddings_dict:
        #             passage_embeddings.update(d)

        #         # order the passage embeddings based on the passage ids
        #         passage_embeddings = dict(
        #             sorted(passage_embeddings.items(), key=lambda x: x[0])
        #         )
        #     else:
        passage_embeddings = passage_embeddings_dict

        # extract the embeddings
        passage_embeddings = list(passage_embeddings.values())
        # stack it
        logger.debug(f"Stacking embeddings at rank {dist.get_rank()}")
        passage_embeddings: torch.Tensor = torch.stack(passage_embeddings, dim=0)

        # move the passage embeddings to the gpu if needed
        if not self.device == "cpu":
            passage_embeddings = passage_embeddings.to(
                PRECISION_MAP[self.precision]
            )
            if self.device != passage_embeddings.device:
                passage_embeddings = passage_embeddings.to(self.device)

            # logger.debug(f"Broadcasting embeddings at rank {dist.get_rank()}")
            # dist.broadcast(passage_embeddings, src=0)
        
        # dist.barrier()
        self.embeddings = passage_embeddings
        del passage_embeddings
        # logger.debug(f"Broadcasting embeddings at rank {dist.get_rank()}")
        # dist.broadcast(self.embeddings, src=0)

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

        with get_autocast_context(self.device, self.embeddings.dtype):
            # move query to the same device as embeddings
            query = query.to(self.embeddings.device)
            if query.dtype != self.embeddings.dtype:
                query = query.to(self.embeddings.dtype)
            similarity = torch.matmul(query, self.embeddings.T)
            # similarity = self.mm(query)
            # Retrieve the indices of the top k passage embeddings
            retriever_out: torch.return_types.topk = torch.topk(
                similarity, k=min(k, similarity.shape[-1]), dim=1
            )

        # get int values
        batch_top_k: List[List[int]] = retriever_out.indices.detach().cpu().tolist()
        # get float values
        batch_scores: List[List[float]] = retriever_out.values.detach().cpu().tolist()
        # Retrieve the passages corresponding to the indices
        batch_docs = [
            [self.get_document_from_index(i) for i in indices]
            for indices in batch_top_k
        ]
        # build the output object
        batch_retrieved_samples = [
            [
                RetrievedSample(document=doc, score=score)
                for doc, score in zip(docs, scores)
            ]
            for docs, scores in zip(batch_docs, batch_scores)
        ]
        return batch_retrieved_samples
