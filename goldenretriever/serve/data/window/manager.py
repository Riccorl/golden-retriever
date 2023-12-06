import collections
import itertools
from typing import Dict, List, Optional, Set, Tuple

from goldenretriever.serve.data.objects import Window
from goldenretriever.serve.data.splitters.base_sentence_splitter import (
    BaseSentenceSplitter,
)
from goldenretriever.serve.data.tokenizers.base_tokenizer import BaseTokenizer


class WindowManager:
    def __init__(
        self, tokenizer: BaseTokenizer, splitter: BaseSentenceSplitter
    ) -> None:
        self.tokenizer = tokenizer
        self.splitter = splitter

    def create_windows(
        self,
        documents: str | List[str],
        window_size: int,
        stride: int,
        max_length: int | None = None,
        doc_topic: str = None,
    ) -> List[Window]:
        """
        Create windows from a list of documents.

        Args:
            documents (:obj:`str` or :obj:`List[str]`):
                The document(s) to split in windows.
            window_size (:obj:`int`):
                The size of the window.
            stride (:obj:`int`):
                The stride between two windows.
            max_length (:obj:`int`, `optional`):
                The maximum length of a window.
            doc_topic (:obj:`str`, `optional`):
                The topic of the document(s).

        Returns:
            :obj:`List[RelikReaderSample]`: The windows created from the documents.
        """
        # normalize input
        if isinstance(documents, str):
            documents = [documents]

        # batch tokenize
        documents_tokens = self.tokenizer(documents)

        # set splitter params
        if hasattr(self.splitter, "window_size"):
            self.splitter.window_size = window_size
        if hasattr(self.splitter, "window_stride"):
            self.splitter.window_stride = stride

        windowed_documents = []
        for doc_id, (document, document_tokens) in enumerate(
            zip(documents, documents_tokens)
        ):
            if doc_topic is None:
                doc_topic = document_tokens[0] if len(document_tokens) > 0 else ""

            splitted_document = self.splitter(document_tokens, max_length=max_length)
            document_windows = []
            for window_id, window in enumerate(splitted_document):
                window_text_start = window[0].idx
                window_text_end = window[-1].idx + len(window[-1].text)
                document_windows.append(
                    Window(
                        doc_id=doc_id,
                        window_id=window_id,
                        text=document[window_text_start:window_text_end],
                        tokens=[w.text for w in window],
                        words=[w.text for w in window],
                        doc_topic=doc_topic,
                        offset=window_text_start,
                        token2char_start={str(i): w.idx for i, w in enumerate(window)},
                        token2char_end={
                            str(i): w.idx + len(w.text) for i, w in enumerate(window)
                        },
                        char2token_start={
                            str(w.idx): w.i for i, w in enumerate(window)
                        },
                        char2token_end={
                            str(w.idx + len(w.text)): w.i for i, w in enumerate(window)
                        },
                    )
                )

            windowed_documents.extend(document_windows)
        return windowed_documents
