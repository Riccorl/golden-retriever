from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from data.labels import Labels


class BaseModel(torch.nn.Module):
    def __init__(self, labels: Labels, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        labels: Optional[torch.Tensor] = None,
        compute_loss: bool = False,
        compute_predictions: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            labels (`torch.Tensor`):
                The labels of the sentences.
            compute_predictions (`bool`):
                Whether to compute the predictions.
            compute_loss (`bool`):
                Whether to compute the loss.

        Returns:
            obj:`torch.Tensor`: The outputs of the model.
        """

        logits = None
        output = {"logits": logits}

        if compute_predictions:
            predictions = logits.argmax(dim=-1)
            output["predictions"] = predictions

        if compute_loss and labels is not None:
            output["loss"] = self.compute_loss(logits, labels)

        return output

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the model.

        Args:
            logits (`torch.Tensor`):
                The logits of the model.
            labels (`torch.Tensor`):
                The labels of the model.

        Returns:
            obj:`torch.Tensor`: The loss of the model.
        """
        # return F.cross_entropy(
        #     logits.view(-1, self.labels.get_label_size()), labels.view(-1)
        # )
        raise NotImplementedError
