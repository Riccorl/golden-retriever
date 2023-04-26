import os
from pathlib import Path
from typing import List, Tuple, Union

import ray
from fastapi import FastAPI, HTTPException
from ray import serve

from golden_retriever import GoldenRetriever
from golden_retriever.common.log import get_console_logger

from ipa.preprocessing.tokenizers.spacy_tokenizer import SpacyTokenizer

from spacy.lang.en import English


logger = get_console_logger()

VERSION = {}  # type: ignore
with open(Path(__file__).parent.parent / "version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# variables
DEVICE = os.environ.get("DEVICE", "cpu")
MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", None)
TOP_K = int(os.environ.get("TOP_K", 100))

app = FastAPI(
    title="Golden Retriever AIDA",
    version=VERSION["VERSION"],
    description="Golden Retriever finetuned on AIDA",
)


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class ModelServer:
    def __init__(self):
        self.retriever = GoldenRetriever.from_pretrained(
            MODEL_NAME_OR_PATH, device=DEVICE
        )
        self.retriever.eval()

        self.tokenizer = SpacyTokenizer(language="en")

    @app.post("/api/retrieve")
    def retrieve_enpoint(self, documents: Union[str, List[str]]):
        # try:
        if isinstance(documents, str):
            documents = [documents]
        return self.retriever.retrieve(documents, k=TOP_K)
        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=f"Server Error: {e}")

    @app.post("/api/gerbil")
    def gerbil_enpoint(self, documents: Union[str, List[str]]):
        def tokenize(
            tokenizer: SpacyTokenizer, document: str
        ) -> Tuple[List[str], List[Tuple[int, int]]]:
            tokenized_document = tokenizer(document)
            tokens = []
            tokens_char_mapping = []
            for token in tokenized_document:
                tokens.append(token.text)
                tokens_char_mapping.append((token.start_char, token.end_char))
            return tokens, tokens_char_mapping

        def split_document_by_window(
            tokenizer: SpacyTokenizer,
            document: str,
            window_size: str,
            stride: int,
            doc_id: int = 0,
            doc_topic: str = None,
        ) -> List[dict]:
            document_tokens, tokens_char_mapping = tokenize(tokenizer, document)
            if doc_topic is None:
                doc_topic = document_tokens[0]
            document_windows = []
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
                    dict(
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

        try:
            # normalize input
            if isinstance(documents, str):
                documents = [documents]

            # output list
            windows_contexts = []

            # split documents into windows
            document_windows = [
                window
                for d_id, d in enumerate(documents)
                for window in split_document_by_window(
                    self.tokenizer, d, window_size=24, stride=12, doc_id=d_id
                )
            ]

            # get text and topic from document windows and create new list
            text = [window["text"] for window in document_windows]
            text_pair = [window["doc_topic"] for window in document_windows]

            # batch retrieval
            batch = []
            for t, t_p in zip(text, text_pair):
                batch.append((t, t_p))
                if len(batch) == 64:
                    t_batch = [t for t, _ in batch]
                    t_p_batch = [t_p for _, t_p in batch]
                    predictions, _ = self.retriever.retrieve(
                        t_batch, t_p_batch, k=TOP_K
                    )
                    windows_contexts.extend(predictions)
                    batch = []

            # leftover batch
            if len(batch) > 0:
                t_batch = [t for t, _ in batch]
                t_p_batch = [t_p for _, t_p in batch]
                predictions, _ = self.retriever.retrieve(t_batch, t_p_batch, k=TOP_K)
                windows_contexts.extend(predictions)

            # add context to document windows
            for window, contexts in zip(document_windows, windows_contexts):
                contexts = [c.split(" <def>", 1)[0] for c in contexts]
                window["window_candidates"] = contexts

            # return document windows
            return document_windows

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Server Error: {e}")


served = ModelServer.bind()
