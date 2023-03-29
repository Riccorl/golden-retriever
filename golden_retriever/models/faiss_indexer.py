from typing import Union

import math
import numpy
import torch

from golden_retriever.common.utils import is_package_available

if is_package_available("faiss"):
    import faiss
    import faiss.contrib.torch_utils


class FaissIndexer:
    def __init__(
        self,
        embeddings: Union[torch.Tensor, numpy.ndarray],
        index: str = "Flat",
        metric: int = "faiss.METRIC_INNER_PRODUCT",
        normalize: bool = False,
        use_gpu: bool = False,
    ) -> None:
        self.use_gpu = use_gpu
        self.normalize = (
            normalize
            and metric == faiss.METRIC_INNER_PRODUCT
            and not isinstance(embeddings, torch.Tensor)
        )
        if self.normalize:
            index = f"L2norm,{index}"
        faiss_vector_size = embeddings.shape[1]
        if not self.use_gpu:
            index = index.replace("x,", "x_HNSW32,")
        index = index.replace("x", str(math.ceil(math.sqrt(faiss_vector_size)) * 4))
        self.index = faiss.index_factory(faiss_vector_size, index, metric)
        # convert to GPU
        if self.use_gpu:
            # use a single GPU
            faiss_resource = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(faiss_resource, 0, self.index)
        else:
            # move to CPU if embeddings is a torch.Tensor
            embeddings = (
                embeddings.cpu() if isinstance(embeddings, torch.Tensor) else embeddings
            )
        # self.index.train(embeddings)
        self.index.add(embeddings)
        # faiss.extract_index_ivf(self.index.quantizer).nprobe = 123

    def search(
        self, query: Union[torch.Tensor, numpy.ndarray], k: int = 1
    ) -> Union[torch.Tensor, numpy.ndarray]:
        k = min(k, self.index.ntotal)
        if self.normalize:
            faiss.normalize_L2(query)
        if isinstance(query, torch.Tensor) and not self.use_gpu:
            query = query.cpu()
        return self.index.search(query, k)[1]
