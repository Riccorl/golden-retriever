from composer.models.base import ComposerModel

from typing import Any, Union

import hydra
import torch
from omegaconf import DictConfig

from goldenretriever.common.model_inputs import ModelInputs


class GoldenRetrieverComposerModule(ComposerModel):
    def __init__(
        self,
        model: Union[torch.nn.Module, DictConfig],
    ):
        super().__init__()
        if isinstance(model, DictConfig):
            self.model = hydra.utils.instantiate(model)
        else:
            self.model = model

    def loss(self, outputs, batch, *args, **kwargs):
        """Accepts the outputs from forward() and the batch"""
        return outputs.loss

    # def get_metrics(self, is_train):
    # return {"MulticlassAccuracy": self.acc}

    def forward(self, batch):
        return self.model(**batch, return_loss=True)

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    # def update_metric(self, batch, outputs, metric) -> None:
    #     _, targets = batch
    #     metric.update(outputs, targets)
