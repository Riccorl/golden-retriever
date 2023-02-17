from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from data.labels import Labels
import transformers as tr

from models.losses import MultiLabelNCELoss


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        language_model: Union[
            str, tr.PreTrainedModel
        ] = "sentence-transformers/all-MiniLM-6-v2",
        *args,
        **kwargs,
    ):
        super().__init__()
        if isinstance(language_model, str):
            self.language_model = tr.AutoModel.from_pretrained(language_model)
        else:
            self.language_model = language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        pooling_strategy: str = "mean",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

        model_output = self.language_model(**model_kwargs)
        if pooling_strategy == "cls":
            return model_output.pooler_output
        elif pooling_strategy == "mean":
            # mean pooling
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            mean_pooling = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return mean_pooling
        else:
            raise ValueError(
                f"Pooling strategy {pooling_strategy} not supported, use 'cls' or 'mean'"
            )


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        question_encoder: SentenceEncoder,
        loss_type: torch.nn.Module,
        context_encoder: Optional[SentenceEncoder] = None,
        labels: Labels = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if labels is not None:
            self.labels = labels

        self.question_encoder = question_encoder

        if not context_encoder:
            context_encoder = question_encoder
        self.context_encoder = context_encoder

        self.loss_type = loss_type

    def forward(
        self,
        questions: Dict[str, torch.Tensor],
        contexts: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False,
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

        Returns:
            obj:`torch.Tensor`: The outputs of the model.
        """
        question_encodings = self.question_encoder(**questions)
        contexts_encodings = self.context_encoder(**contexts)
        logits = torch.matmul(question_encodings, contexts_encodings.T)

        output = {"logits": logits}

        if return_loss and labels is not None:
            output["loss"] = self.loss_type(logits, labels)

        return output
