from typing import Union

import faiss
import faiss.contrib.torch_utils
import numpy
import torch


class FaissIndexer:
    def __init__(
        self,
        embeddings: Union[torch.Tensor, numpy.ndarray],
        index: str = "Flat",
        metric: int = faiss.METRIC_INNER_PRODUCT,
        normalize: bool = True,
        use_gpu: bool = False,
    ) -> None:
        self.normalize = (
            normalize
            and metric == faiss.METRIC_INNER_PRODUCT
            and not isinstance(embeddings, torch.Tensor)
        )
        if self.normalize:
            faiss.normalize_L2(embeddings)
        self.index = faiss.index_factory(embeddings.shape[1], index, metric)
        # convert to GPU
        self.use_gpu = use_gpu
        if self.use_gpu:
            # use a single GPU
            faiss_resource = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(faiss_resource, 0, self.index)
        else:
            # move to CPU if embeddings is a torch.Tensor
            embeddings = (
                embeddings.cpu() if isinstance(embeddings, torch.Tensor) else embeddings
            )
        self.index.add(embeddings)

    def search(
        self, query: Union[torch.Tensor, numpy.ndarray], k: int = 1
    ) -> Union[torch.Tensor, numpy.ndarray]:
        k = min(k, self.index.ntotal)
        if self.normalize:
            faiss.normalize_L2(query)
        if isinstance(query, torch.Tensor) and not self.use_gpu:
            query = query.cpu()
        return self.index.search(query, k)[1]
