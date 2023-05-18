import os


class ServerParameterManager:
    def __init__(self) -> None:
        self.device = os.environ.get("DEVICE", "cpu")
        self.index_device = os.environ.get("INDEX_DEVICE", self.device)
        self.precision = os.environ.get("PRECISION", "fp32")
        self.index_precision = os.environ.get("INDEX_PRECISION", self.precision)
        self.model_name_or_path = os.environ.get("MODEL_NAME_OR_PATH", None)
        self.top_k = int(os.environ.get("TOP_K", 100))
        self.use_faiss = os.environ.get("USE_FAISS", False)
        self.window_batch_size = int(os.environ.get("WINDOW_BATCH_SIZE", 32))
        self.window_size = int(os.environ.get("WINDOW_SIZE", 32))
        self.window_stride = int(os.environ.get("WINDOW_SIZE", 16))
        self.split_on_spaces = os.environ.get("SPLIT_ON_SPACES", False)


class RayParameterManager:
    def __init__(self) -> None:
        self.num_gpus = int(os.environ.get("NUM_GPUS", 1))
        self.min_replicas = int(os.environ.get("MIN_REPLICAS", 1))
        self.max_replicas = int(os.environ.get("MAX_REPLICAS", 1))
