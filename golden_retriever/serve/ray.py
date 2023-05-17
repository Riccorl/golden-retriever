import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from fastapi import FastAPI
from golden_retriever.serve.tokenizers import (
    WhitespaceTokenizer,
    SpacyTokenizer,
    RegexTokenizer,
)
from ray import serve

from golden_retriever import GoldenRetriever
from golden_retriever.common.log import get_console_logger, get_logger
from golden_retriever.serve.window.manager import WindowManager
from golden_retriever.data.utils import batch_generator

logger = get_logger(__name__, level=logging.INFO)
console_logger = get_console_logger()

VERSION = {}  # type: ignore
with open(Path(__file__).parent.parent / "version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# variables
DEVICE = os.environ.get("DEVICE", "cpu")
INDEX_DEVICE = os.environ.get("INDEX_DEVICE", DEVICE)
PRECISION = os.environ.get("PRECISION", "fp32")
INDEX_PRECISION = os.environ.get("INDEX_PRECISION", PRECISION)
MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", None)
TOP_K = int(os.environ.get("TOP_K", 100))
USE_FAISS = os.environ.get("USE_FAISS", False)
WINDOW_BATCH_SIZE = int(os.environ.get("WINDOW_BATCH_SIZE", 32))
SPLIT_ON_SPACES = os.environ.get("SPLIT_ON_SPACES", False)
NUM_GPUS = int(os.environ.get("NUM_GPUS", 1))
MIN_REPLICAS = int(os.environ.get("MIN_REPLICAS", 1))
MAX_REPLICAS = int(os.environ.get("MAX_REPLICAS", 1))

app = FastAPI(
    title="Golden Retriever",
    version=VERSION["VERSION"],
    description="Golden Retriever REST API",
)


@serve.deployment(
    ray_actor_options={"num_gpus": NUM_GPUS if DEVICE == "cuda" else 0},
    autoscaling_config={"min_replicas": MIN_REPLICAS, "max_replicas": MAX_REPLICAS},
)
@serve.ingress(app)
class GoldenRetrieverServer:
    def __init__(self):
        # check that the model exists
        if not os.path.exists(MODEL_NAME_OR_PATH):
            raise ValueError(
                f"Model {MODEL_NAME_OR_PATH} does not exist. Please specify a valid path."
            )

        # log stuff for debugging
        logger.info(f"MODEL_NAME_OR_PATH: {MODEL_NAME_OR_PATH}")
        logger.info(f"TOP_K: {TOP_K}")
        logger.info(f"DEVICE: {DEVICE}")
        logger.info(f"INDEX_DEVICE: {INDEX_DEVICE}")
        logger.info(f"PRECISION: {PRECISION}")
        logger.info(f"INDEX_PRECISION: {INDEX_PRECISION}")
        logger.info(f"USE_FAISS: {USE_FAISS}")
        logger.info(f"WINDOW_BATCH_SIZE: {WINDOW_BATCH_SIZE}")
        logger.info(f"SPLIT_ON_SPACES: {SPLIT_ON_SPACES}")

        self.retriever = GoldenRetriever.from_pretrained(
            MODEL_NAME_OR_PATH,
            device=DEVICE,
            index_device=INDEX_DEVICE,
            index_precision=INDEX_PRECISION,
            load_faiss_index=USE_FAISS,
        )
        self.retriever.eval()

        if SPLIT_ON_SPACES:
            # self.tokenizer = WhitespaceTokenizer()
            self.tokenizer = RegexTokenizer()
        else:
            self.tokenizer = SpacyTokenizer(language="en")

        self.window_manager = WindowManager(tokenizer=self.tokenizer)

    @app.post("/api/retrieve")
    def retrieve_endpoint(
        self,
        documents: Union[str, List[str]],
        document_topics: Optional[Union[str, List[str]]] = None,
    ):
        # try:
        if isinstance(documents, str):
            documents = [documents]
        if document_topics is not None:
            if isinstance(document_topics, str):
                document_topics = [document_topics]
            assert len(documents) == len(document_topics)
        return self.retriever.retrieve(
            documents, text_pair=document_topics, k=TOP_K, precision=PRECISION
        )
        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=f"Server Error: {e}")

    @app.post("/api/gerbil")
    def gerbil_endpoint(self, documents: Union[str, List[str]]):
        # normalize input
        if isinstance(documents, str):
            documents = [documents]

        # output list
        windows_contexts = []
        # split documents into windows
        document_windows = [
            window
            for d_id, d in enumerate(documents)
            for window in self.window_manager(
                self.tokenizer, d, window_size=24, stride=12, doc_id=d_id
            )
        ]

        # get text and topic from document windows and create new list
        model_inputs = [
            tuple(window.text, window.doc_topic) for window in document_windows
        ]

        # batch generator
        for batch in batch_generator(model_inputs, batch_size=WINDOW_BATCH_SIZE):
            text, text_pair = zip(*batch)
            batch_predictions = self.retriever.retrieve(
                text, text_pair, k=TOP_K, precision=PRECISION
            )
            windows_contexts.extend(
                [[p.label for p in predictions] for predictions in batch_predictions]
            )

        # add context to document windows
        for window, contexts in zip(document_windows, windows_contexts):
            # clean up contexts (remove everything after first <def> tag if present)
            contexts = [c.split(" <def>", 1)[0] for c in contexts]
            window["window_candidates"] = contexts

        # return document windows
        return document_windows

        # except Exception as e:
        #     logger.error(f"Server Error: {e}")
        #     raise HTTPException(status_code=500, detail=f"Server Error: {e}")


server = GoldenRetrieverServer.bind()
