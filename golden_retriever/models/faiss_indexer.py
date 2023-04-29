import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint

from golden_retriever.common.log import get_console_logger
from golden_retriever.common.utils import is_package_available

if is_package_available("faiss"):
    import faiss
    import faiss.contrib.torch_utils

logger = get_console_logger()


@dataclass
class FaissOutput:
    indices: Union[torch.Tensor, numpy.ndarray]
    distances: Union[torch.Tensor, numpy.ndarray]


class FaissIndexer:
    def __init__(
        self,
        index=None,
        embeddings: Union[torch.Tensor, numpy.ndarray] = None,
        index_type: str = "Flat",
        metric: int = faiss.METRIC_INNER_PRODUCT,
        normalize: bool = False,
        use_gpu: bool = False,
    ) -> None:
        if embeddings is not None and index is not None:
            logger.log("Both embeddings and index are provided, ignoring embeddings.")

        if embeddings is None and index is None:
            raise ValueError("Either embeddings or index must be provided.")

        self.use_gpu = use_gpu

        if index is not None:
            self.index = index
            if self.use_gpu:
                # use a single GPU
                faiss_resource = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(faiss_resource, 0, self.index)
        else:
            self.normalize = (
                normalize
                and metric == faiss.METRIC_INNER_PRODUCT
                and not isinstance(embeddings, torch.Tensor)
            )
            if self.normalize:
                index_type = f"L2norm,{index_type}"
            faiss_vector_size = embeddings.shape[1]
            if not self.use_gpu:
                index_type = index_type.replace("x,", "x_HNSW32,")
            index_type = index_type.replace(
                "x", str(math.ceil(math.sqrt(faiss_vector_size)) * 4)
            )
            self.index = faiss.index_factory(faiss_vector_size, index_type, metric)

            # convert to GPU
            if self.use_gpu:
                # use a single GPU
                faiss_resource = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(faiss_resource, 0, self.index)
            else:
                # move to CPU if embeddings is a torch.Tensor
                embeddings = (
                    embeddings.cpu()
                    if isinstance(embeddings, torch.Tensor)
                    else embeddings
                )
            # self.index.train(embeddings)
            self.index.add(embeddings)
            # faiss.extract_index_ivf(self.index.quantizer).nprobe = 123

            # save parameters for saving/loading
            self.index_type = index_type
            self.metric = metric

    def reconstruct(self, index: int) -> Union[torch.Tensor, numpy.ndarray]:
        return self.index.reconstruct(index)

    def search(
        self, query: Union[torch.Tensor, numpy.ndarray], k: int = 1
    ) -> FaissOutput:
        k = min(k, self.index.ntotal)
        if self.normalize:
            faiss.normalize_L2(query)
        if isinstance(query, torch.Tensor) and not self.use_gpu:
            query = query.cpu()
        query_results = self.index.search(query, k)
        return FaissOutput(indices=query_results[1], distances=query_results[0])

    def save(self, saving_dir: Union[str, os.PathLike]):
        saving_dir = Path(saving_dir)
        faiss_index_path = saving_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(faiss_index_path))
        config_path = saving_dir / "faiss_config.json"
        config = {
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize": self.normalize,
            "use_gpu": self.use_gpu,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, loading_dir: Union[str, os.PathLike], **kwargs) -> "FaissIndexer":
        loading_dir = Path(loading_dir)
        config_path = loading_dir / "faiss_config.json"
        faiss_index_path = loading_dir / "faiss_index.bin"
        index = faiss.read_index(faiss_index_path)
        with open(config_path, "r") as f:
            config = json.load(f)
        config.update({"index": index})
        # kwargs can overwrite config
        config.update(kwargs)
        logger.log(f"Loading FaissIndexer from {loading_dir}")
        logger.log(f"Faiss Config:")
        pprint(OmegaConf.to_container(config), console=logger, expand_all=True)
        return cls(**config)
