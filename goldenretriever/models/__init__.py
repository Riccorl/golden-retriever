from dataclasses import dataclass
import torch


PRECISION_MAP = {
    None: torch.float32,
    16: torch.float16,
    32: torch.float32,
    "float16": torch.float16,
    "float32": torch.float32,
    "half": torch.float16,
    "float": torch.float32,
    "16": torch.float16,
    "32": torch.float32,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass
class RetrievedSample:
    """
    Dataclass for the output of the GoldenRetriever model.
    """

    score: float
    index: int
    label: str
