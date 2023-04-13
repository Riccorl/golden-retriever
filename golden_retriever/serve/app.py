import os
from ast import Tuple
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException

from golden_retriever import GoldenRetriever
from golden_retriever.common.logging import get_console_logger

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
retriever = GoldenRetriever.from_pretrained(MODEL_NAME_OR_PATH, device=DEVICE)
retriever.eval()


@app.post("/api/retrieve")
def read_request(user_request_in: List[str]):
    try:
        return retriever.retrieve(user_request_in, k=TOP_K)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")
