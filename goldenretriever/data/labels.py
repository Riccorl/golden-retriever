import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import transformers as tr


class Labels:
    """
    Class that contains the labels for a model.

    Args:
        _labels_to_index (:obj:`Dict[str, Dict[str, int]]`):
            A dictionary from :obj:`str` to :obj:`int`.
        _index_to_labels (:obj:`Dict[str, Dict[int, str]]`):
            A dictionary from :obj:`int` to :obj:`str`.
    """

    def __init__(
        self,
        _labels_to_index: Dict[str, Dict[str, int]] = None,
        _index_to_labels: Dict[str, Dict[int, str]] = None,
        **kwargs,
    ):
        self._labels_to_index = _labels_to_index or {}
        self._index_to_labels = _index_to_labels or {}
        # if _labels_to_index is not empty and _index_to_labels is not provided
        # to the constructor, build the inverted label dictionary
        if not _index_to_labels and _labels_to_index:
            for namespace in self._labels_to_index:
                self._index_to_labels[namespace] = {
                    v: k for k, v in self._labels_to_index[namespace].items()
                }

    def get_index_from_label(self, label: str, namespace: str = "labels") -> int:
        """
        Returns the index of a literal label.

        Args:
            label (:obj:`str`):
                The string representation of the label.
            namespace (:obj:`str`, optional, defaults to ``labels``):
                The namespace where the label belongs, e.g. ``roles`` for a SRL task.

        Returns:
            :obj:`int`: The index of the label.
        """
        if namespace not in self._labels_to_index:
            raise ValueError(
                f"Provided namespace `{namespace}` is not in the label dictionary."
            )

        if label not in self._labels_to_index[namespace]:
            raise ValueError(f"Provided label {label} is not in the label dictionary.")

        return self._labels_to_index[namespace][label]

    def get_label_from_index(self, index: int, namespace: str = "labels") -> str:
        """
        Returns the string representation of the label index.

        Args:
            index (:obj:`int`):
                The index of the label.
            namespace (:obj:`str`, optional, defaults to ``labels``):
                The namespace where the label belongs, e.g. ``roles`` for a SRL task.

        Returns:
            :obj:`str`: The string representation of the label.
        """
        if namespace not in self._index_to_labels:
            raise ValueError(
                f"Provided namespace `{namespace}` is not in the label dictionary."
            )

        if index not in self._index_to_labels[namespace]:
            raise ValueError(
                f"Provided label `{index}` is not in the label dictionary."
            )

        return self._index_to_labels[namespace][index]

    def add_labels(
        self,
        labels: Union[str, List[str], Set[str], Dict[str, int]],
        namespace: str = "labels",
    ) -> List[int]:
        """
        Adds the labels in input in the label dictionary.

        Args:
            labels (:obj:`str`, :obj:`List[str]`, :obj:`Set[str]`):
                The labels (single label, list of labels or set of labels) to add to the dictionary.
            namespace (:obj:`str`, optional, defaults to ``labels``):
                Namespace where the labels belongs.

        Returns:
            :obj:`List[int]`: The index of the labels just inserted.
        """
        if isinstance(labels, dict):
            self._labels_to_index[namespace] = labels
            self._index_to_labels[namespace] = {
                v: k for k, v in self._labels_to_index[namespace].items()
            }
        # normalize input
        if isinstance(labels, (str, list)):
            labels = set(labels)
        # if new namespace, add to the dictionaries
        if namespace not in self._labels_to_index:
            self._labels_to_index[namespace] = {}
            self._index_to_labels[namespace] = {}
        # returns the new indices
        return [self._add_label(label, namespace) for label in labels]

    def _add_label(self, label: str, namespace: str = "labels") -> int:
        """
        Adds the label in input in the label dictionary.

        Args:
            label (:obj:`str`):
                The label to add to the dictionary.
            namespace (:obj:`str`, optional, defaults to ``labels``):
                Namespace where the label belongs.

        Returns:
            :obj:`List[int]`: The index of the label just inserted.
        """
        if label not in self._labels_to_index[namespace]:
            index = len(self._labels_to_index[namespace])
            self._labels_to_index[namespace][label] = index
            self._index_to_labels[namespace][index] = label
            return index
        else:
            return self._labels_to_index[namespace][label]

    def get_labels(self, namespace: str = "labels") -> Dict[str, int]:
        """
        Returns all the labels that belongs to the input namespace.

        Args:
            namespace (:obj:`str`, optional, defaults to ``labels``):
                Labels namespace to retrieve.

        Returns:
            :obj:`Dict[str, int]`: The label dictionary, from ``str`` to ``int``.
        """
        if namespace not in self._labels_to_index:
            raise ValueError(
                f"Provided namespace `{namespace}` is not in the label dictionary."
            )
        return self._labels_to_index[namespace]

    def get_label_size(self, namespace: str = "labels") -> int:
        """
        Returns the number of the labels in the namespace dictionary.

        Args:
            namespace (:obj:`str`, optional, defaults to ``labels``):
                Labels namespace to retrieve.

        Returns:
            :obj:`int`: Number of labels.
        """
        if namespace not in self._labels_to_index:
            raise ValueError(
                f"Provided namespace `{namespace}` is not in the label dictionary."
            )
        return len(self._labels_to_index[namespace])

    def get_namespaces(self) -> List[str]:
        """
        Returns all the namespaces in the label dictionary.

        Returns:
            :obj:`List[str]`: The namespaces in the label dictionary.
        """
        return list(self._labels_to_index.keys())

    @classmethod
    def from_file(cls, file_path: Union[str, Path, dict], **kwargs):
        with open(file_path, "r") as f:
            labels_to_index = json.load(f)
        return cls(labels_to_index, **kwargs)

    def save(self, file_path: Union[str, Path, dict], **kwargs):
        with open(file_path, "w") as f:
            json.dump(self._labels_to_index, f, indent=2)


class PassageManager:
    def __init__(
        self,
        tokenizer: Optional[tr.PreTrainedTokenizer] = None,
        passages: Optional[Union[Dict[str, Dict[str, int]], Labels, List[str]]] = None,
        lazy: bool = True,
        **kwargs,
    ):
        if passages is None:
            self.passages = Labels()
        elif isinstance(passages, Labels):
            self.passages = passages
        elif isinstance(passages, dict):
            self.passages = Labels(passages)
        elif isinstance(passages, list):
            self.passages = Labels()
            self.passages.add_labels(passages)
        else:
            raise ValueError(
                "`passages` should be either a Labels object or a dictionary."
            )

        self.tokenizer = tokenizer
        self.lazy = lazy

        self._tokenized_passages = {}

        if not self.lazy:
            self._tokenize_passages(self.passages)

    def __len__(self) -> int:
        return self.passages.get_label_size()

    def get_index_from_passage(self, passage: str) -> int:
        """
        Returns the index of the passage in input.

        Args:
            passage (:obj:`str`):
                The passage to get the index from.

        Returns:
            :obj:`int`: The index of the passage.
        """
        return self.passages.get_index_from_label(passage)

    def get_passage_from_index(self, index: int) -> str:
        """ "
        Returns the passage from the index in input.

        Args:
            index (:obj:`int`):
                The index to get the passage from.

        Returns:
            :obj:`str`: The passage.
        """
        return self.passages.get_label_from_index(index)

    def add_passages(
        self,
        passages: Union[str, List[str], Set[str], Dict[str, int]],
        lazy: Optional[bool] = None,
    ) -> List[int]:
        """
        Adds the passages in input in the passage dictionary.

        Args:
            passages (:obj:`str`, :obj:`List[str]`, :obj:`Set[str]`, :obj:`Dict[str, int]`):
                The passages (single passage, list of passages, set of passages or dictionary of passages) to add to the dictionary.
            lazy (:obj:`bool`, optional, defaults to ``None``):
                Whether to tokenize the passages right away or not.

        Returns:
            :obj:`List[int]`: The index of the passages just inserted.
        """

        return self.passages.add_labels(passages)

    def get_passages(self) -> Dict[str, int]:
        """
        Returns all the passages in the passage dictionary.

        Returns:
            :obj:`Dict[str, int]`: The passage dictionary, from ``str`` to ``int``.
        """
        return self.passages.get_labels()

    def get_tokenized_passage(
        self, passage: Union[str, int], force_tokenize: bool = False, **kwargs
    ) -> Dict:
        """
        Returns the tokenized passage in input.

        Args:
            passage (:obj:`Union[str, int]`):
                The passage to tokenize.
            force_tokenize (:obj:`bool`, optional, defaults to ``False``):
                Whether to force the tokenization of the passage or not.
            kwargs:
                Additional keyword arguments to pass to the tokenizer.

        Returns:
            :obj:`Dict`: The tokenized passage.
        """
        passage_index: int | None = None
        passage_str: str | None = None

        if isinstance(passage, str):
            passage_index = self.passages.get_index_from_label(passage)
            passage_str = passage
        elif isinstance(passage, int):
            passage_index = passage
            passage_str = self.passages.get_label_from_index(passage)
        else:
            raise ValueError(
                f"`passage` should be either a `str` or an `int`. Provided type: {type(passage)}."
            )

        if passage_index not in self._tokenized_passages or force_tokenize:
            self._tokenized_passages[passage_index] = self.tokenizer(
                passage_str, **kwargs
            )

        return self._tokenized_passages[passage_index]

    def _tokenize_passages(self, **kwargs):
        for passage in self.passages.get_labels():
            self.get_tokenized_passage(passage, **kwargs)

    def tokenize(self, text: Union[str, List[str]], **kwargs):
        """
        Tokenizes the text in input using the tokenizer.

        Args:
            text (:obj:`str`, :obj:`List[str]`):
                The text to tokenize.
            **kwargs:
                Additional keyword arguments to pass to the tokenizer.

        Returns:
            :obj:`List[str]`: The tokenized text.

        """
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer was provided. Please provide a tokenizer to the passageManager."
            )
        return self.tokenizer(text, **kwargs)
