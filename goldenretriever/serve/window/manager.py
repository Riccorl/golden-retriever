from dataclasses import dataclass
from typing import List, Optional, Tuple
from goldenretriever.serve.tokenizers.base_tokenizer import BaseTokenizer


@dataclass
class Window:
    doc_id: int
    window_id: int
    text: str
    tokens: List[str]
    doc_topic: Optional[str]
    offset: int
    token2char_start: dict
    token2char_end: dict
    window_candidates: Optional[List[str]] = None


class WindowManager:
    def __init__(self, tokenizer: BaseTokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(
        self, tokenizer: BaseTokenizer, document: str
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        tokenized_document = tokenizer(document)
        tokens = []
        tokens_char_mapping = []
        for token in tokenized_document:
            tokens.append(token.text)
            tokens_char_mapping.append((token.start_char, token.end_char))
        return tokens, tokens_char_mapping

    def __call__(
        self,
        tokenizer: BaseTokenizer,
        document: str,
        window_size: int,
        stride: int,
        doc_id: int = 0,
        doc_topic: str = None,
    ) -> List[Window]:
        document_tokens, tokens_char_mapping = self.tokenize(tokenizer, document)
        if doc_topic is None:
            doc_topic = document_tokens[0] if len(document_tokens) > 0 else ""
        document_windows = []
        if len(document_tokens) <= window_size:
            text = document
            document_windows.append(
                Window(
                    doc_id=doc_id,
                    window_id=0,
                    text=text,
                    tokens=document_tokens,
                    doc_topic=doc_topic,
                    offset=0,
                    token2char_start={
                        i: tokens_char_mapping[i][0]
                        for i in range(len(document_tokens))
                    },
                    token2char_end={
                        i: tokens_char_mapping[i][1]
                        for i in range(len(document_tokens))
                    },
                )
            )
        else:
            for window_id, i in enumerate(range(0, len(document_tokens), stride)):
                # if the last stride is smaller than the window size, then we can
                # include more tokens form the previous window.
                if i != 0 and i + window_size > len(document_tokens):
                    overflowing_tokens = i + window_size - len(document_tokens)
                    if overflowing_tokens >= stride:
                        break
                    i -= overflowing_tokens

                involved_token_indices = list(
                    range(i, min(i + window_size, len(document_tokens) - 1))
                )
                window_tokens = [document_tokens[j] for j in involved_token_indices]
                window_text_start = tokens_char_mapping[involved_token_indices[0]][0]
                window_text_end = tokens_char_mapping[involved_token_indices[-1]][1]
                text = document[window_text_start:window_text_end]
                document_windows.append(
                    Window(
                        doc_id=doc_id,
                        window_id=window_id,
                        text=text,
                        tokens=window_tokens,
                        doc_topic=doc_topic,
                        offset=window_text_start,
                        token2char_start={
                            i: tokens_char_mapping[ti][0]
                            for i, ti in enumerate(involved_token_indices)
                        },
                        token2char_end={
                            i: tokens_char_mapping[ti][1]
                            for i, ti in enumerate(involved_token_indices)
                        },
                    )
                )
        return document_windows
