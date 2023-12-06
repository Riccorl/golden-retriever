import logging
from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from ray_old import serve

from goldenretriever import GoldenRetriever
from goldenretriever.common.log import get_logger
from goldenretriever.data.utils import batch_generator
from goldenretriever.serve.tokenizers import SpacyTokenizer, WhitespaceTokenizer
from goldenretriever.serve.utils import RayParameterManager, ServerParameterManager
from goldenretriever.serve.window.manager import WindowManager

logger = get_logger(__name__, level=logging.INFO)

VERSION = {}  # type: ignore
with open(Path(__file__).parent.parent.parent / "version.py", "r") as version_file:
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
    def __init__(
        self,
        question_encoder: str,
        document_index: str,
        passage_encoder: str | None = None,
        top_k: int = 100,
        device: str = "cpu",
        index_device: str | None = None,
        precision: int = 32,
        index_precision: int | None = None,
        use_faiss: bool = False,
        window_batch_size: int = 32,
        window_size: int = 32,
        window_stride: int = 16,
        split_on_spaces: bool = False,
    ):
        # parameters
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder
        self.document_index = document_index
        self.top_k = top_k
        self.device = device
        self.index_device = index_device or device
        self.precision = precision
        self.index_precision = index_precision or precision
        self.use_faiss = use_faiss
        self.window_batch_size = window_batch_size
        self.window_size = window_size
        self.window_stride = window_stride
        self.split_on_spaces = split_on_spaces

        # log stuff for debugging
        logger.info("Initializing GoldenRetrieverServer with parameters:")
        logger.info(f"QUESTION_ENCODER: {self.question_encoder}")
        logger.info(f"PASSAGE_ENCODER: {self.passage_encoder}")
        logger.info(f"DOCUMENT_INDEX: {self.document_index}")
        logger.info(f"TOP_K: {self.top_k}")
        logger.info(f"DEVICE: {self.device}")
        logger.info(f"INDEX_DEVICE: {self.index_device}")
        logger.info(f"PRECISION: {self.precision}")
        logger.info(f"INDEX_PRECISION: {self.index_precision}")
        logger.info(f"WINDOW_BATCH_SIZE: {self.window_batch_size}")
        logger.info(f"SPLIT_ON_SPACES: {self.split_on_spaces}")

        self.retriever = GoldenRetriever(
            question_encoder=self.question_encoder,
            passage_encoder=self.passage_encoder,
            document_index=self.document_index,
            device=self.device,
            index_device=self.index_device,
            index_precision=self.index_precision,
        )
        self.retriever.eval()

        if self.split_on_spaces:
            logger.info("Using WhitespaceTokenizer")
            self.tokenizer = WhitespaceTokenizer()
            # logger.info("Using RegexTokenizer")
            # self.tokenizer = RegexTokenizer()
        else:
            logger.info("Using SpacyTokenizer")
            self.tokenizer = SpacyTokenizer(language="en")

        self.window_manager = WindowManager(tokenizer=self.tokenizer)

    # @serve.batch()
    async def handle_batch(
        self, documents: List[str], document_topics: List[str]
    ) -> List:
        return self.retriever.retrieve(
            documents, text_pair=document_topics, k=self.top_k, precision=self.precision
        )

    @app.post("/api/retrieve")
    async def retrieve_endpoint(
        self,
        documents: Union[str, List[str]],
        document_topics: Optional[Union[str, List[str]]] = None,
    ):
        try:
            # normalize input
            if isinstance(documents, str):
                documents = [documents]
            if document_topics is not None:
                if isinstance(document_topics, str):
                    document_topics = [document_topics]
                assert len(documents) == len(document_topics)
            # get predictions
            return await self.handle_batch(documents, document_topics)
        except Exception as e:
            # log the entire stack trace
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Server Error: {e}")

    @app.post("/api/gerbil")
    async def gerbil_endpoint(self, documents: Union[str, List[str]]):
        try:
            # normalize input
            if isinstance(documents, str):
                documents = [documents]

            # output list
            windows_passages = []
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
            model_inputs = [
                (window.text, window.doc_topic) for window in document_windows
            ]

            # batch generator
            for batch in batch_generator(
                model_inputs, batch_size=self.window_batch_size
            ):
                text, text_pair = zip(*batch)
                batch_predictions = await self.handle_batch(text, text_pair)
                windows_passages.extend(
                    [
                        [p.label for p in predictions]
                        for predictions in batch_predictions
                    ]
                )

            # add passage to document windows
            for window, passages in zip(document_windows, windows_passages):
                # clean up passages (remove everything after first <def> tag if present)
                passages = [c.split(" <def>", 1)[0] for c in passages]
                window.window_candidates = passages

            # return document windows
            return document_windows

        except Exception as e:
            # log the entire stack trace
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Server Error: {e}")


server = GoldenRetrieverServer.bind(**vars(SERVER_MANAGER))
