import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import numpy
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint

from goldenretriever.common.log import get_console_logger, get_logger
from goldenretriever.common.utils import (
    from_cache,
    is_remote_url,
    is_str_a_path,
    relative_to_absolute_path,
    sapienzanlp_model_urls,
)
from goldenretriever.data.labels import Labels

# from goldenretriever.models.model import GoldenRetriever, RetrievedSample


logger = get_logger(__name__)
console_logger = get_console_logger()


@dataclass
class IndexerOutput:
    indices: Union[torch.Tensor, numpy.ndarray]
    distances: Union[torch.Tensor, numpy.ndarray]


class BaseDocumentIndex:
    CONFIG_NAME = "config.json"
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

            self.documents = Labels()
            self.documents.add_labels(documents)

        self.embeddings = embeddings

    @property
    def config(self) -> Dict[str, Any]:
        """
        The configuration of the document index.

        Returns:
            `Dict[str, Any]`: The configuration of the retriever.
        """

        def obj_to_dict(obj):
            match obj:
                case dict():
                    data = {}
                    for k, v in obj.items():
                        data[k] = obj_to_dict(v)
                    return data

                case list() | tuple():
                    return [obj_to_dict(x) for x in obj]

                case object(__dict__=_):
                    data = {
                        "_target_": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                    }
                    for k, v in obj.__dict__.items():
                        if not k.startswith("_"):
                            data[k] = obj_to_dict(v)
                    return data

                case _:
                    return obj

        return obj_to_dict(self)

    def index(
        self,
        retriever,
        *args,
        **kwargs,
    ) -> "BaseDocumentIndex":
        raise NotImplementedError

    def search(self, query: Any, k: int = 1, *args, **kwargs) -> List:
        raise NotImplementedError

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

    def save_pretrained(
        self,
        output_dir: Union[str, os.PathLike],
        config: Optional[Dict[str, Any]] = None,
        config_file_name: Optional[str] = None,
        document_file_name: Optional[str] = None,
        embedding_file_name: Optional[str] = None,
    ):
        """
        Save the retriever to a directory.

        Args:
            output_dir (`str`):
                The directory to save the retriever to.
            config (`Optional[Dict[str, Any]]`, `optional`):
                The configuration to save. If `None`, the current configuration of the retriever will be
                saved. Defaults to `None`.
        """
        if config is None:
            # create a default config
            config = self.config

        config_file_name = config_file_name or self.CONFIG_NAME
        document_file_name = document_file_name or self.DOCUMENTS_FILE_NAME
        embedding_file_name = embedding_file_name or self.EMBEDDINGS_FILE_NAME

        # create the output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving retriever to {output_dir}")
        logger.info(f"Saving config to {output_dir / config_file_name}")
        # pretty print the config
        pprint(config, console=console_logger, expand_all=True)
        OmegaConf.save(config, output_dir / config_file_name)

        # save the current state of the retriever
        embedding_path = output_dir / embedding_file_name
        logger.info(f"Saving retriever state to {output_dir / embedding_path}")
        torch.save(self.embeddings, embedding_path)

        # save the passage index
        documents_path = output_dir / document_file_name
        logger.info(f"Saving passage index to {documents_path}")
        self.documents.save(documents_path)

        logger.info("Saving document index to disk done.")

    @classmethod
    def from_pretrained(
        cls,
        index_name_or_dir: Union[str, os.PathLike],
        device: str = "cpu",
        precision: Optional[str] = None,
        config_file_name: Optional[str] = None,
        document_file_name: Optional[str] = None,
        embedding_file_name: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "BaseDocumentIndex":
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)

        if is_remote_url(index_name_or_dir):
            # if model_name_or_dir is a URL
            # download it and try to load
            model_archive = index_name_or_dir
        elif Path(index_name_or_dir).is_dir() or Path(index_name_or_dir).is_file():
            # if model_name_or_dir is a local directory or
            # an archive file try to load it
            model_archive = index_name_or_dir
        else:
            # probably model_name_or_dir is a sapienzanlp model id
            # guess the url and try to download
            model_name_or_dir_ = index_name_or_dir
            # raise ValueError(f"Providing a model id is not supported yet.")
            model_archive = sapienzanlp_model_urls(model_name_or_dir_)

        config_file_name = config_file_name or cls.EMBEDDINGS_FILE_NAME
        document_file_name = document_file_name or cls.DOCUMENTS_FILE_NAME
        embedding_file_name = embedding_file_name or cls.EMBEDDINGS_FILE_NAME

        model_dir = from_cache(
            model_archive,
            filenames=[config_file_name, document_file_name, embedding_file_name],
            cache_dir=cache_dir,
            force_download=force_download,
        )

        config_path = model_dir / config_file_name
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found at {config_path}."
            )
        config = OmegaConf.load(config_path)
        pprint(OmegaConf.to_container(config), console=console_logger, expand_all=True)

        # load the documents
        documents_path = model_dir / document_file_name

        if not documents_path.exists():
            raise ValueError(f"Document file `{documents_path}` does not exist.")
        logger.info(f"Loading documents from {documents_path}")
        documents = Labels.from_file(documents_path)

        # load the passage embeddings
        embedding_path = model_dir / embedding_file_name
        # run some checks
        if embedding_path.exists():
            logger.info(f"Loading embeddings from {embedding_path}")
            embeddings = torch.load(embedding_path, map_location="cpu")
        else:
            logger.warning(f"Embedding file `{embedding_path}` does not exist.")

        document_index = hydra.utils.instantiate(
            config,
            documents=documents,
            embeddings=embeddings,
            device=device,
            precision=precision,
            *args,
            **kwargs,
        )

        return document_index
        # return cls(
        #     documents=documents,
        #     embeddings=embeddings,
        #     device=device,
        #     precision=precision,
        #     **kwargs,
        # )
