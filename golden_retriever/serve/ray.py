import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from fastapi import FastAPI
from ray import serve

from golden_retriever import GoldenRetriever
from golden_retriever.common.log import get_console_logger, get_logger
from golden_retriever.data.utils import batch_generator
from golden_retriever.serve.tokenizers import (
    RegexTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)
from golden_retriever.serve.utils import RayParameterManager, ServerParameterManager
from golden_retriever.serve.window.manager import WindowManager

logger = get_logger(__name__, level=logging.INFO)

VERSION = {}  # type: ignore
with open(Path(__file__).parent.parent / "version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# Env variables for server
SERVER_MANAGER = ServerParameterManager()
RAY_MANAGER = RayParameterManager()

app = FastAPI(
    title="Golden Retriever",
    version=VERSION["VERSION"],
    description="Golden Retriever REST API",
)


@serve.deployment(
    ray_actor_options={
        "num_gpus": RAY_MANAGER.num_gpus if SERVER_MANAGER.device == "cuda" else 0
    },
    autoscaling_config={
        "min_replicas": RAY_MANAGER.min_replicas,
        "max_replicas": RAY_MANAGER.max_replicas,
    },
)
@serve.ingress(app)
class GoldenRetrieverServer:
    def __init__(self):
        # parameters
        self.model_name_or_path = SERVER_MANAGER.model_name_or_path
        self.top_k = SERVER_MANAGER.top_k
        self.device = SERVER_MANAGER.device
        self.index_device = SERVER_MANAGER.index_device
        self.precision = SERVER_MANAGER.precision
        self.index_precision = SERVER_MANAGER.index_precision
        self.use_faiss = SERVER_MANAGER.use_faiss
        self.window_batch_size = SERVER_MANAGER.window_batch_size
        self.window_size = SERVER_MANAGER.window_size
        self.window_stride = SERVER_MANAGER.window_stride
        self.split_on_spaces = SERVER_MANAGER.split_on_spaces

        # check that the model exists
        if not os.path.exists(self.model_name_or_path):
            raise ValueError(
                f"Model {self.model_name_or_path} does not exist. Please specify a valid path."
            )

        # log stuff for debugging
        logger.info(f"MODEL_NAME_OR_PATH: {self.model_name_or_path}")
        logger.info(f"TOP_K: {self.top_k}")
        logger.info(f"DEVICE: {self.device}")
        logger.info(f"INDEX_DEVICE: {self.index_device}")
        logger.info(f"PRECISION: {self.precision}")
        logger.info(f"INDEX_PRECISION: {self.index_precision}")
        logger.info(f"USE_FAISS: {self.use_faiss}")
        logger.info(f"WINDOW_BATCH_SIZE: {self.window_batch_size}")
        logger.info(f"SPLIT_ON_SPACES: {self.split_on_spaces}")
        logger.info(f"SPLIT_ON_SPACES: {self.split_on_spaces}")

        self.retriever = GoldenRetriever.from_pretrained(
            self.model_name_or_path,
            device=self.device,
            index_device=self.index_device,
            index_precision=self.index_precision,
            load_faiss_index=self.use_faiss,
        )
        self.retriever.eval()

        if self.split_on_spaces:
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
            documents, text_pair=document_topics, k=self.top_k, precision=self.precision
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
            for doc_id, document in enumerate(documents)
            for window in self.window_manager(
                self.tokenizer,
                document,
                window_size=self.window_size,
                stride=self.window_stride,
                doc_id=doc_id,
            )
        ]

        # get text and topic from document windows and create new list
        model_inputs = [(window.text, window.doc_topic) for window in document_windows]

        # batch generator
        for batch in batch_generator(model_inputs, batch_size=self.window_batch_size):
            text, text_pair = zip(*batch)
            batch_predictions = self.retriever.retrieve(
                text, text_pair, k=self.top_k, precision=self.precision
            )
            windows_contexts.extend(
                [[p.label for p in predictions] for predictions in batch_predictions]
            )

        # add context to document windows
        for window, contexts in zip(document_windows, windows_contexts):
            # clean up contexts (remove everything after first <def> tag if present)
            contexts = [c.split(" <def>", 1)[0] for c in contexts]
            window.window_candidates = contexts

        # return document windows
        return document_windows

        # except Exception as e:
        #     logger.error(f"Server Error: {e}")
        #     raise HTTPException(status_code=500, detail=f"Server Error: {e}")


server = GoldenRetrieverServer.bind()
