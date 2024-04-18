from typing import Any, List, Union

import hydra
import lightning as pl
import torch
from omegaconf import DictConfig

from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.utils import HardNegativesManagerThread


class GoldenRetrieverPLModule(pl.LightningModule):
    def __init__(
        self,
        model: Union[torch.nn.Module, DictConfig],
        optimizer: Union[torch.optim.Optimizer, DictConfig],
        lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, DictConfig] = None,
        micro_batch_size: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        if isinstance(model, DictConfig):
            self.model = hydra.utils.instantiate(model)
        else:
            self.model = model

        self.hn_algo = HardNegativeAlgorithm()

        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.micro_batch_size = micro_batch_size

    def forward(self, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        return self.model(**kwargs)

    def training_step(self, batch: ModelInputs, batch_idx: int) -> torch.Tensor:
        batch = self.hn_algo(batch, self)
        batch = self.trainer.train_dataloader.collate_fn.split_batch(
            batch, self.micro_batch_size
        )
        loss = self._training_step(batch)
        # forward_output = self.forward(**batch, return_loss=True)
        return loss

    def _training_step(
        self, batches: List[ModelInputs], batch_idx: int
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for batch in batches:
            forward_output = self.forward(**batch, return_loss=True)
            loss += forward_output["loss"]
            self.log(
                "loss",
                forward_output["loss"],
                batch_size=batch["questions"]["input_ids"].size(0),
                prog_bar=True,
                sync_dist=True,
            )
            # log the passage batch size
            self.log(
                "passage_batch_size",
                batch["passages"]["input_ids"].size(0),
                batch_size=batch["questions"]["input_ids"].size(0),
                prog_bar=True,
                sync_dist=True,
            )
        return loss / len(batches)

    def validation_step(self, batch: ModelInputs, batch_idx: int) -> None:
        forward_output = self.forward(**batch, return_loss=True)
        self.log(
            "val_loss",
            forward_output["loss"],
            batch_size=batch["questions"]["input_ids"].size(0),
            sync_dist=True,
        )

    def test_step(self, batch: ModelInputs, batch_idx: int) -> Any:
        forward_output = self.forward(**batch, return_loss=True)
        self.log(
            "test_loss",
            forward_output["loss"],
            batch_size=batch["questions"]["input_ids"].size(0),
            sync_dist=True,
        )

    def configure_model(self):
        if self.model is not None:
            return

    def configure_optimizers(self):
        if isinstance(self.optimizer_config, DictConfig):
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if "layer_norm_layer" in n
                    ],
                    "weight_decay": self.hparams.optimizer.weight_decay,
                    "lr": 1e-4,
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if all(nd not in n for nd in no_decay)
                        and "layer_norm_layer" not in n
                    ],
                    "weight_decay": self.hparams.optimizer.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if "layer_norm_layer" not in n
                        and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = hydra.utils.instantiate(
                self.optimizer_config,
                # params=self.parameters(),
                params=optimizer_grouped_parameters,
                _convert_="partial",
            )
        else:
            optimizer = self.optimizer_config

        if self.lr_scheduler_config is None:
            return optimizer

        if isinstance(self.lr_scheduler_config, DictConfig):
            lr_scheduler = hydra.utils.instantiate(
                self.lr_scheduler_config, optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler_config

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]


class HardNegativeAlgorithm:

    def __init__(self, tokenizer=None, max_length: int = None):
        if tokenizer is not None and max_length is not None:
            self.hn_manager = HardNegativesManagerThread(
                tokenizer, max_length=max_length
            )
        else:
            # we don't have the tokenizer and the max_length yet
            # delay the initialization of the hn_manager
            self.hn_manager = None

    def __call__(self, batch, pl_module: GoldenRetrieverPLModule) -> None:
        try:
            self.hn_manager = HardNegativesManagerThread()
        except TypeError:
            # a little hack to avoid the initialization of the hn_manager
            # without the tokenizer and the max_length
            return batch

        sample_idxs = batch["sample_idx"]
        hn_passages = {}
        i = 0
        for sample in sample_idxs:
            if sample in self.hn_manager:
                i += 1
                hn_passages.update(
                    {
                        tuple(passage["input_ids"]): passage
                        for passage in self.hn_manager.get(sample)
                    }
                )

        # if there are no hard negatives, return
        if len(hn_passages) == 0:
            return batch

        # get dataloader collator
        collator = pl_module.trainer.train_dataloader.collate_fn
        hn_passages = list(hn_passages.values())
        hn_passages_batch = ModelInputs(collator.convert_to_batch(hn_passages))
        hn_passages_batch = hn_passages_batch.to(pl_module.device)
        # get the questions
        questions = batch["questions"]
        # get the passages
        passages = batch["passages"]
        # build an index to map the position of the passage in the batch
        passage_index = {tuple(c["input_ids"]): i for i, c in enumerate(hn_passages)}

        # now we can create the labels
        labels = torch.zeros(
            questions["input_ids"].shape[0], hn_passages_batch["input_ids"].shape[0]
        )
        labels = labels.to(batch.labels.device)
        # iterate over the questions and set the labels to 1 if the passage is positive
        for sample_idx in range(len(questions["input_ids"])):
            for pssg in batch["positives_pssgs"][sample_idx]:
                # get the index of the positive passage
                index = passage_index.get(tuple(pssg["input_ids"]), None)
                # set the label to 1
                if index is not None:
                    labels[sample_idx, index] = 1

        # now concatenate the passages and the hard negatives
        passages_ids = torch.cat(
            [batch.passages["input_ids"], hn_passages_batch["input_ids"]], dim=0
        )
        # concatenate the attention masks
        attention_mask = torch.cat(
            [batch.passages["attention_mask"], hn_passages_batch["attention_mask"]],
            dim=0,
        )
        # concatenate the token type ids
        token_type_ids = torch.cat(
            [batch.passages["token_type_ids"], hn_passages_batch["token_type_ids"]],
            dim=0,
        )
        # concatenate the labels
        labels = torch.cat([batch.labels, labels], dim=1)
        # update the batch
        passages = {
            "input_ids": passages_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        batch["passages"] = passages
        batch["labels"] = labels

        return batch

    @staticmethod
    def duplicate(tensor_one: torch.Tensor, tensor_two: torch.Tensor) -> torch.Tensor:
        """
        Check if two tensors have the same elements.

        Args:
            tensor_one (`torch.Tensor`): The first tensor.
            tensor_two (`torch.Tensor`): The second tensor.

        Returns:
            `torch.Tensor`: A boolean tensor with the same shape as the input tensors.
        """
        # dimensions
        shape1 = tensor_one.shape[0]
        shape2 = tensor_two.shape[0]
        c = tensor_one.shape[1]
        assert c == tensor_two.shape[1], "Tensors must have same number of columns"

        a_expand = tensor_one.unsqueeze(1).expand(-1, shape2, c)
        b_expand = tensor_two.unsqueeze(0).expand(shape1, -1, c)
        # element-wise equality
        mask = (a_expand == b_expand).all(-1).any(-1)
        return mask
