from __future__ import annotations

import logging

import torch
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

from goldenretriever.data.utils import HardNegativesManager

log = logging.getLogger(__name__)


class HardNegativeAlgorithm(Algorithm):

    def __init__(
        self,
        tokenizer,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hn_manager = HardNegativesManager(tokenizer, max_length=max_length)

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.BEFORE_FORWARD]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        # update hn_manager
        # self.hn_manager = HardNegativesManager(
        #     self.tokenizer, max_length=self.max_length
        # )

        # get the hard negatives
        batch = state.batch
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
            return

        # get dataloader collator
        collator = state.train_dataloader.collate_fn
        hn_passages = list(hn_passages.values())
        hn_passages_batch = collator.convert_to_batch(hn_passages)
        hn_passages_batch = state.device.batch_to_device(hn_passages_batch)
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

        # update the batch
        state.batch_set_item("passages", passages)
        state.batch_set_item("labels", labels)

    @staticmethod
    def duplicate(tensor_one: torch.Tenser, tensor_two: torch.Tenser) -> torch.Tensor:
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
