import json
from pathlib import Path
from typing import Dict, List, Union

from relik.common.log import get_logger

logger = get_logger(__name__)


class Document:
    def __init__(
        self,
        text: str,
        id: int | None = None,
        metadata: Dict | None = None,
        **kwargs,
    ):
        self.text = text
        # if id is not provided, we use the hash of the text
        self.id = id if id is not None else hash(text)
        # if metadata is not provided, we use an empty dictionary
        self.metadata = metadata or {}

    def __str__(self):
        return f"{self.id}:{self.text}"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {"text": self.text, "id": self.id, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        with open(file_path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def save(self, file_path: Union[str, Path], **kwargs):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class DocumentStore:
    """
    A document store is a collection of documents.

    Args:
        documents (:obj:`List[Document]`):
            The documents to store.
    """

    def __init__(self, documents: List[Document] = None) -> None:
        if documents is None:
            documents = []
        self._documents = documents
        # build an index for the documents
        self._documents_index = {doc.id: doc for doc in self._documents}
        # build a reverse index for the documents
        self._documents_reverse_index = {doc.text: doc for doc in self._documents}

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index):
        return self._documents[index]

    def __iter__(self):
        return iter(self._documents)

    def __contains__(self, item):
        return item in self._documents_index

    def __str__(self):
        return f"DocumentStore with {len(self)} documents"

    def __repr__(self):
        return self.__str__()

    def get_document_from_id(self, id: int) -> Document | None:
        """
        Retrieve a document by its ID.

        Args:
            id (`int`):
                The ID of the document to retrieve.

        Returns:
            Optional[Document]: The document with the given ID, or None if it does not exist.
        """
        if id not in self._documents_index:
            logger.warning(f"Document with id `{id}` does not exist, skipping")
        return self._documents_index.get(id, None)

    def get_document_from_text(self, text: str) -> Document | None:
        """
        Retrieve the document by its text.

        Args:
            text (`str`):
                The text of the document to retrieve.

        Returns:
            Optional[Document]: The document with the given text, or None if it does not exist.
        """
        return self._documents_reverse_index.get(text, None)

    def add_document(
        self, text: str, id: int | None = None, metadata: Dict | None = None
    ) -> Document:
        """
        Add a document to the document store.

        Args:
            text (`str`):
                The text of the document to add.
            id (`int`, optional, defaults to None):
                The ID of the document to add.
            metadata (`Dict`, optional, defaults to None):
                The metadata of the document to add.

        Returns:
            Document: The document just added.
        """
        if id is None:
            id = hash(text)
        if id in self._documents_index:
            logger.warning(f"Document with id `{id}` already exists, skipping")
            return self._documents_index[id]
        self._documents_index[id] = Document(text, id, metadata)
        self._documents_reverse_index[text] = Document(text, id, metadata)
        return self._documents_index[id]

    def delete_document(self, document: int | str | Document) -> bool:
        """
        Delete a document from the document store.

        Args:
            document (`int`, `str` or `Document`):
                The document to delete.

        Returns:
            bool: True if the document has been deleted, False otherwise.
        """
        if isinstance(document, int):
            return self.delete_by_id(document)
        elif isinstance(document, str):
            return self.delete_by_text(document)
        elif isinstance(document, Document):
            return self.delete_by_document(document)
        else:
            raise ValueError(
                f"Document must be an int, a str or a Document, got `{type(document)}`"
            )

    def delete_by_id(self, id: int) -> bool:
        """
        Delete a document by its ID.

        Args:
            id (`int`):
                The ID of the document to delete.

        Returns:
            bool: True if the document has been deleted, False otherwise.
        """
        if id not in self._documents_index:
            logger.warning(f"Document with id `{id}` does not exist, skipping")
            return False
        del self._documents_reverse_index[self._documents_index[id]]
        del self._documents_index[id]
        return True

    def delete_by_text(self, text: str) -> bool:
        """
        Delete a document by its text.

        Args:
            text (`str`):
                The text of the document to delete.

        Returns:
            bool: True if the document has been deleted, False otherwise.
        """
        if text not in self._documents_reverse_index:
            logger.warning(f"Document with text `{text}` does not exist, skipping")
            return False
        del self._documents_reverse_index[text]
        del self._documents_index[self._documents_index[text]]
        return True

    def delete_by_document(self, document: Document) -> bool:
        """
        Delete a document by its text.

        Args:
            document (:obj:`Document`):
                The document to delete.

        Returns:
            bool: True if the document has been deleted, False otherwise.
        """
        if document.id not in self._documents_index:
            logger.warning(f"Document {document} does not exist, skipping")
            return False
        del self._documents_index[document.id]
        del self._documents_reverse_index[self._documents_index[document.id]]

    def to_dict(self):
        return [doc.to_dict() for doc in self._documents]

    @classmethod
    def from_dict(cls, d):
        return cls([Document.from_dict(doc) for doc in d])

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        with open(file_path, "r") as f:
            # load a json lines file
            d = [Document.from_dict(json.loads(line)) for line in f]
        return cls(d)

    def save(self, file_path: Union[str, Path], **kwargs):
        with open(file_path, "w") as f:
            for doc in self._documents:
                # save as json lines
                f.write(json.dumps(doc.to_dict()) + "\n")
