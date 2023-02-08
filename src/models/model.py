from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from data.labels import Labels
import transformers as tr


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        question_encoder: Union[str, tr.PreTrainedModel] = "bert-base-cased",
        context_encoder: Union[str, tr.PreTrainedModel] = "bert-base-cased",
        labels: Labels = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if labels is not None:
            self.labels = labels
        if isinstance(question_encoder, str):
            self.question_encoder = tr.AutoModel.from_pretrained(question_encoder)
        else:
            self.question_encoder = question_encoder
        if isinstance(context_encoder, str):
            self.context_encoder = tr.AutoModel.from_pretrained(context_encoder)
        else:
            self.context_encoder = context_encoder

    @staticmethod
    def encode(
        encoder: tr.PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids
        return encoder(**model_kwargs).pooler_output

    def forward(
        self,
        questions: Dict[str, torch.Tensor],
        contexts: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        return_predictions: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            questions (`Dict[str, torch.Tensor]`):
                The questions to encode.
            contexts (`Dict[str, torch.Tensor]`):
                The contexts to encode.
            labels (`torch.Tensor`):
                The labels of the sentences.
            return_loss (`bool`):
                Whether to compute the predictions.
            return_predictions (`bool`):
                Whether to compute the loss.

        Returns:
            obj:`torch.Tensor`: The outputs of the model.
        """
        question_encodings = self.encode(self.question_encoder, **questions)
        contexts_encodings = self.encode(self.context_encoder, **contexts)
        logits = torch.matmul(question_encodings, contexts_encodings.T)

        if len(question_encodings.size()) > 1:
            q_num = question_encodings.size(0)
            logits = logits.view(q_num, -1)

        logits = F.log_softmax(logits, dim=1)

        output = {"logits": logits}

        if return_predictions:
            _, predictions = torch.max(logits, 1)
            output["predictions"] = predictions

        if return_loss and labels is not None:
            output["loss"] = self.compute_loss(logits, labels)

        return output

    @staticmethod
    def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
        return F.nll_loss(logits, labels, reduction="mean")
