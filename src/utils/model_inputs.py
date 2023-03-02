from __future__ import annotations

from collections import UserDict
from typing import Any, Union

import torch

from src.utils.logging import get_console_logger

logger = get_console_logger()


class ModelInputs(UserDict):
    """Model input dictionary wrapper."""

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError(f"`ModelInputs` has no attribute `{item}`")

    def __getitem__(self, item: str) -> Any:
        return self.data[item]

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    def keys(self):
        """A set-like object providing a view on D's keys."""
        return self.data.keys()

    def values(self):
        """An object providing a view on D's values."""
        return self.data.values()

    def items(self):
        """A set-like object providing a view on D's items."""
        return self.data.items()

    def to(self, device: Union[str, torch.device]) -> ModelInputs:
        """
        Send all tensors values to device.
        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            :class:`tokenizers.ModelInputs`: The same instance of :class:`~tokenizers.ModelInputs`
            after modification.
        """
        if isinstance(device, (str, torch.device, int)):
            self.data = {
                k: v.to(device=device) if hasattr(v, "to") else v
                for k, v in self.data.items()
            }
        else:
            logger.log(
                f"Attempting to cast to another type, {str(device)}. This is not supported."
            )
        return self
