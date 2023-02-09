from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from data.labels import Labels
import transformers as tr


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        language_model: Union[str, tr.PreTrainedModel] = "bert-base-cased",
        question_encoder: Optional[Union[str, tr.PreTrainedModel]] = None,
        context_encoder: Optional[Union[str, tr.PreTrainedModel]] = None,
        loss_type: str = "nll",
        labels: Labels = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if labels is not None:
            self.labels = labels

        if not question_encoder:
            question_encoder = language_model
        if isinstance(question_encoder, str):
            self.question_encoder = tr.AutoModel.from_pretrained(question_encoder)
        else:
            self.question_encoder = question_encoder
        if not context_encoder:
            context_encoder = language_model
        if isinstance(context_encoder, str):
            self.context_encoder = tr.AutoModel.from_pretrained(context_encoder)
        else:
            self.context_encoder = context_encoder

        self.loss_type = loss_type

    def encode(
        self,
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
        return self.mean_pooling(encoder(**model_kwargs), attention_mask)

    @staticmethod
    def mean_pooling(
        model_output: tr.modeling_utils.ModelOutput, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

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
            # _, predictions = torch.max(logits, 1)
            predictions = torch.sigmoid(logits)
            predictions[predictions >= 0.5] = 1
            output["predictions"] = predictions

        if return_loss and labels is not None:
            output["loss"] = self.compute_loss(logits, labels)

        return output

    # @staticmethod
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
        # return F.nll_loss(logits, labels, reduction="mean")
        return self.sum_log_nce_loss(logits, labels, reduction="sum")

    def sum_log_nce_loss(self, logits, mask, reduction="sum"):
        """
        :param logits: reranking logits(B x C) or span loss(B x C x L)
        :param mask: reranking mask(B x C) or span mask(B x C x L)
        :return: sum log p_positive i  over (positive i, negatives)
        """
        gold_scores = logits.masked_fill(~(mask.bool()), 0)
        gold_scores_sum = gold_scores.sum(-1)  # B x C
        neg_logits = logits.masked_fill(mask.bool(), float("-inf"))  # B x C x L
        neg_log_sum_exp = torch.logsumexp(neg_logits, -1, keepdim=True)  # B x C x 1
        norm_term = (
            torch.logaddexp(logits, neg_log_sum_exp)
            .masked_fill(~(mask.bool()), 0)
            .sum(-1)
        )
        gold_log_probs = gold_scores_sum - norm_term
        loss = -gold_log_probs.sum()
        if reduction == "mean":
            print("mean reduction")
            loss /= logits.size(0)
        return loss
