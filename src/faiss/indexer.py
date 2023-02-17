from typing import Union
import faiss
import numpy
import torch


class FaissIndexer:
    def __init__(
        self,
        embeddings: Union[torch.Tensor, numpy.ndarray],
        index: str = "Flat",
        metric: int = 0,  # faiss.METRIC_INNER_PRODUCT,
        normalize: bool = True,
    ) -> None:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        self.normalize = normalize
        if normalize:
            faiss.normalize_L2(embeddings)
        self.index = faiss.index_factory(embeddings.shape[1], index, metric)
        self.index.add(embeddings)

    def search(
        self, query: Union[torch.Tensor, numpy.ndarray], k: int = 1
    ) -> numpy.ndarray:
        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()
        if self.index.metric_type == 0 and self.normalize:
            faiss.normalize_L2(query)
        return self.index.search(query, k)[1]
