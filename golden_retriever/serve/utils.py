from dataclasses import dataclass
import os
from typing import Union

@dataclass
class ServerParameterManager:
    device: str = os.environ.get("DEVICE", "cpu")
    index_device: str = os.environ.get("INDEX_DEVICE", device)
    precision: Union[str, int] = os.environ.get("PRECISION", "fp32")
    index_precision: Union[str, int] = os.environ.get("INDEX_PRECISION", precision)
    model_name_or_path: str = os.environ.get("MODEL_NAME_OR_PATH", None)
    top_k: int = int(os.environ.get("TOP_K", 100))
    use_faiss: bool = os.environ.get("USE_FAISS", False)
    window_batch_size: int = int(os.environ.get("WINDOW_BATCH_SIZE", 32))
    window_size: int = int(os.environ.get("WINDOW_SIZE", 32))
    window_stride: int = int(os.environ.get("WINDOW_SIZE", 16))
    split_on_spaces: bool = os.environ.get("SPLIT_ON_SPACES", False)


class RayParameterManager:
    def __init__(self) -> None:
        self.num_gpus = int(os.environ.get("NUM_GPUS", 1))
        self.min_replicas = int(os.environ.get("MIN_REPLICAS", 1))
        self.max_replicas = int(os.environ.get("MAX_REPLICAS", 1))
