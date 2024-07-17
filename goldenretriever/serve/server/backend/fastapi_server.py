from argparse import ArgumentParser
import logging
import os
from pathlib import Path
from typing import List, Union

import psutil
import torch
import uvicorn

from goldenretriever.common.utils import is_package_available
from goldenretriever.pytorch_modules.model import GoldenRetriever

if not is_package_available("fastapi"):
    raise ImportError(
        "FastAPI is not installed. Please install FastAPI with `pip install goldenretriever[serve]`."
    )
from fastapi import APIRouter, FastAPI, HTTPException

from goldenretriever.common.log import get_logger
from goldenretriever.serve.server.backend.utils import (
    RayParameterManager,
    ServerParameterManager,
)

import fastapi_cli.cli as fastapi_cli

logger = get_logger(__name__)

VERSION = {}  # type: ignore
with open(
    Path(__file__).parent.parent.parent.parent / "version.py", "r"
) as version_file:
    exec(version_file.read(), VERSION)

# Env variables for server
SERVER_MANAGER = ServerParameterManager()
RAY_MANAGER = RayParameterManager()


class GoldenRetrieverServer:
    def __init__(
        self,
        question_encoder: str | None = None,
        passage_encoder: str | None = None,
        index: str | None = None,
        device: str = "cpu",
        index_device: str | None = None,
        precision: str | int | torch.dtype = 32,
        index_precision: str | int | torch.dtype | None = None,
        top_k: int = 100,
        **kwargs,
    ):
        num_threads = os.getenv("TORCH_NUM_THREADS", psutil.cpu_count(logical=False))
        torch.set_num_threads(num_threads)
        logger.info(f"Torch is running on {num_threads} threads.")
        # parameters
        logger.info(f"QUESTION_ENCODER: {question_encoder}")
        self.question_encoder = question_encoder
        logger.info(f"PASSAGE_ENCODER: {passage_encoder}")
        self.passage_encoder = passage_encoder
        logger.info(f"INDEX: {index}")
        self.index = index
        logger.info(f"DEVICE: {device}")
        self.device = device
        if index_device is not None:
            logger.info(f"INDEX_DEVICE: {index_device}")
        self.index_device = index_device or device
        logger.info(f"PRECISION: {precision}")
        self.precision = precision
        if index_precision is not None:
            logger.info(f"INDEX_PRECISION: {index_precision}")
        self.index_precision = index_precision or precision
        logger.info(f"TOP_K: {top_k}")
        self.top_k = top_k

        self.retriever = GoldenRetriever(
            question_encoder=self.question_encoder,
            passage_encoder=self.passage_encoder,
            document_index=self.index,
            device=self.device,
            index_device=self.index_device,
            precision=self.precision,
            index_precision=self.index_precision,
        )

        self.router = APIRouter()
        self.router.add_api_route(
            "/api/goldenretriever", self.endpoint, methods=["POST"]
        )

        logger.info("RelikServer initialized.")

    # @serve.batch()
    async def __call__(self, text: List[str], top_k: int) -> List:
        return self.retriever.retrieve(text, top_k=top_k)

    # @app.post("/api/goldenretriever")
    async def endpoint(self, text: Union[str, List[str]], top_k: int | None = None):
        try:
            top_k = top_k or self.top_k
            # get predictions for the retriever
            return await self(text, top_k)
        except Exception as e:
            # log the entire stack trace
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Server Error: {e}")


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--question-encoder", type=str, required=True)
    arg_parser.add_argument("--passage-encoder", type=str, default=None)
    arg_parser.add_argument("--index", type=str, required=True)
    arg_parser.add_argument("--device", type=str, default="cpu")
    arg_parser.add_argument("--index-device", type=str, default=None)
    arg_parser.add_argument("--precision", type=str, default=32)
    arg_parser.add_argument("--index-precision", type=str, default=None)
    arg_parser.add_argument("--top-k", type=int, default=100)
    arg_parser.add_argument("--host", type=str, default="0.0.0.0")
    arg_parser.add_argument("--port", type=int, default=8000)
    args = arg_parser.parse_args()

    app = FastAPI(
        title="Golden Retriever",
        version=VERSION["VERSION"],
        description="Golden Retriever REST API",
    )
    server = GoldenRetrieverServer(**vars(args))
    app.include_router(server.router)
    # fastapi_cli.run(path=Path(__file__), app=app)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
