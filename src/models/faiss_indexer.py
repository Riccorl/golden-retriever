from typing import Union

import numpy
import torch

import faiss
import faiss.contrib.torch_utils


class FaissIndexer:
    def __init__(
        self,
        embeddings: Union[torch.Tensor, numpy.ndarray],
        index: str = "Flat",
        metric: int = faiss.METRIC_INNER_PRODUCT,
        normalize: bool = True,
        use_gpu: bool = False,
    ) -> None:
        # if isinstance(embeddings, torch.Tensor):
        #     embeddings = embeddings.cpu().numpy()
        self.normalize = normalize
        self.normalize_condition = (
            normalize
            and metric == faiss.METRIC_INNER_PRODUCT
            and not isinstance(embeddings, torch.Tensor)
        )
        if self.normalize_condition:
            faiss.normalize_L2(embeddings)
        self.index = faiss.index_factory(embeddings.shape[1], index, metric)
        # convert to GPU
        if use_gpu:
            # use a single GPU
            faiss_resource = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(faiss_resource, 0, self.index)
        self.index.add(embeddings)

    def search(
        self, query: Union[torch.Tensor, numpy.ndarray], k: int = 1
    ) -> numpy.ndarray:
        k = min(k, self.index.ntotal)
        # if isinstance(query, torch.Tensor):
        #     query = query.cpu().numpy()
        if self.normalize_condition:
            faiss.normalize_L2(query)
        return self.index.search(query, k)[1]
