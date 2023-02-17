from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import transformers as tr
from torch.optim import RAdam, AdamW

from data.labels import Labels


class BasePLModule(pl.LightningModule):
    def __init__(
        self, model: torch.nn.Module, labels: Labels = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.labels = labels
        self.model = model

    def forward(self, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        return self.model(**kwargs)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**{**batch, "return_loss": True})
        self.log("loss", forward_output["loss"])
        return forward_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**{**batch, "return_loss": True})
        self.log("val_loss", forward_output["loss"])

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        forward_output = self.forward(**{**batch, "return_loss": True})
        self.log("test_loss", forward_output["loss"])

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        if self.hparams.optim_params.optimizer in ["radam", "adam"]:
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.optim_params.weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.hparams.optim_params.optimizer == "radam":
                optimizer = RAdam(
                    optimizer_grouped_parameters, lr=self.hparams.optim_params.lr
                )
            elif self.hparams.optim_params.optimizer == "adam":
                optimizer = AdamW(
                    optimizer_grouped_parameters, lr=self.hparams.optim_params.lr
                )
            else:
                raise ValueError(
                    f"Unknown optimizer {self.hparams.optim_params.optimizer}"
                )
        elif self.hparams.optim_params.optimizer == "fuseadam":
            try:
                from deepspeed.ops.adam import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install DeepSpeed (`pip install deepspeed`) to use FuseAdam optimizer."
                )

            optimizer = FusedAdam(self.parameters())
        elif self.hparams.optim_params.optimizer == "deepspeedcpuadam":
            try:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
            except ImportError:
                raise ImportError(
                    "Please install DeepSpeed (`pip install deepspeed`) to use DeepSpeedCPUAdam optimizer."
                )

            optimizer = DeepSpeedCPUAdam(self.parameters())
        elif self.hparams.optim_params.optimizer == "adafactor":
            optimizer = tr.Adafactor(
                self.parameters(),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=self.hparams.optim_params.lr,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.hparams.optim_params.optimizer}")

        if self.hparams.optim_params.use_scheduler:
            lr_scheduler = tr.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.optim_params.num_warmup_steps,
                num_training_steps=self.hparams.optim_params.num_training_steps,
            )
            return [optimizer], [lr_scheduler]

        return optimizer
