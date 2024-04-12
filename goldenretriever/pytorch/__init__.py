from dataclasses import dataclass

import torch

from goldenretriever.indexers.document import Document

@dataclass
class RetrievedSample:
    """
    Dataclass for the output of the GoldenRetriever model.
    """

    score: float
    document: Document
