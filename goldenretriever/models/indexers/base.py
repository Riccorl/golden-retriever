import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy
import torch
from torch.utils.data import DataLoader
import tqdm

from goldenretriever.common.log import get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.common.utils import is_str_a_path, relative_to_absolute_path
from goldenretriever.data.base.datasets import BaseDataset
from goldenretriever.data.labels import Labels
from goldenretriever.models import PRECISION_MAP

# from goldenretriever.models.model import GoldenRetriever, RetrievedSample


logger = get_logger(__name__)


@dataclass
class IndexerOutput:
    indices: Union[torch.Tensor, numpy.ndarray]
    distances: Union[torch.Tensor, numpy.ndarray]


class BaseDocumentIndex:
    DOCUMENTS_FILE_NAME = "documents.json"
    EMBEDDINGS_FILE_NAME = "embeddings.pt"

    def __init__(
        self,
        documents: Union[str, List[str], Labels, os.PathLike, List[os.PathLike]],
        embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        if isinstance(documents, Labels):
            self.documents = documents
        else:
            documents_are_paths = False

            # normalize the documents to list if not already
            if not isinstance(documents, list):
                documents = [documents]

            # now check if the documents are a list of paths (either str or os.PathLike)
            if isinstance(documents[0], str) or isinstance(documents[0], os.PathLike):
                # check if the str is a path
                documents_are_paths = is_str_a_path(documents[0])

            # if the documents are a list of paths, then we load them
            if documents_are_paths:
                logger.info("Loading documents from paths")
                _documents = []
                for doc in documents:
                    with open(relative_to_absolute_path(doc)) as f:
                        _documents += [line.strip() for line in f.readlines()]
                # remove duplicates
                documents = list(set(_documents))

        # # documents to be used for indexing
        # if isinstance(documents, Labels):
        #     self.documents = documents
        # else:
            self.documents = Labels()
            self.documents.add_labels(documents)

        self.embeddings = embeddings

    def index(
        self,
        retriever,
        *args,
        **kwargs,
    ) -> "BaseDocumentIndex":
        raise NotImplementedError

    def search(self, query: Any, k: int = 1, *args, **kwargs) -> List:
        raise NotImplementedError

    # @property
    # def embeddings(self) -> torch.Tensor:
    #     """
    #     The document embeddings.
    #     """
    #     return self.embeddings

    # @property
    # def documents(self) -> Labels:
    #     """
    #     The document labels.
    #     """
    #     return self.documents

    def get_index_from_passage(self, document: str) -> int:
        """
        Get the index of the passage.

        Args:
            passage (`str`):
                The document to get the index for.

        Returns:
            `int`: The index of the document.
        """
        return self.documents.get_index_from_label(document)

    def get_passage_from_index(self, index: int) -> str:
        """
        Get the document from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `str`: The document.
        """
        return self.documents.get_label_from_index(index)

    def get_embeddings_from_index(self, index: int) -> torch.Tensor:
        """
        Get the document vector from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.embeddings is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        if index >= self.embeddings.shape[0]:
            raise ValueError(
                f"The index {index} is out of bounds. The maximum index is {len(self.embeddings) - 1}."
            )
        return self.embeddings[index]

    def get_embeddings_from_passage(self, document: str) -> torch.Tensor:
        """
        Get the document vector from the document label.

        Args:
            document (`str`):
                The document to get the vector for.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.embeddings is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        return self.get_embeddings_from_index(self.get_index_from_passage(document))

    def save(self, saving_dir: Union[str, os.PathLike]):
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        loading_dir: Union[str, os.PathLike],
        *args,
        **kwargs,
    ) -> "BaseDocumentIndex":
        raise NotImplementedError
