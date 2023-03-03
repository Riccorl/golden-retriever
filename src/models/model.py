from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
import transformers as tr

from data.labels import Labels


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        language_model: Union[
            str, tr.PreTrainedModel
        ] = "sentence-transformers/all-MiniLM-6-v2",
        pooling_strategy: str = "mean",
        *args,
        **kwargs,
    ):
        super().__init__()
        if isinstance(language_model, str):
            self.language_model = tr.AutoModel.from_pretrained(language_model)
        else:
            self.language_model = language_model
        self.pooling_strategy = pooling_strategy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        pooling_strategy: Optional[str] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if pooling_strategy is None:
            pooling_strategy = self.pooling_strategy

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


class GoldenRetriever(torch.nn.Module):
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

        # question encoder model
        self.question_encoder = question_encoder
        if not context_encoder:
            # if no context encoder is provided, s
            # hare the weights of the question encoder
            context_encoder = question_encoder
        # context encoder model
        self.context_encoder = context_encoder
        # loss function
        self.loss_type = loss_type

        # indexer stuff
        self.indexer = None

    def forward(
        self,
        questions: Dict[str, torch.Tensor] = None,
        contexts: Dict[str, torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        question_encodings: Optional[torch.Tensor] = None,
        contexts_encodings: Optional[torch.Tensor] = None,
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
            question_encodings (`torch.Tensor`):
                The encodings of the questions.
            contexts_encodings (`torch.Tensor`):
                The encodings of the contexts.

        Returns:
            obj:`torch.Tensor`: The outputs of the model.
        """
        if questions is None and question_encodings is None:
            raise ValueError(
                "Either `questions` or `question_encodings` must be provided"
            )
        if contexts is None and contexts_encodings is None:
            raise ValueError(
                "Either `contexts` or `contexts_encodings` must be provided"
            )

        if question_encodings is None:
            question_encodings = self.question_encoder(**questions)
        if contexts_encodings is None:
            contexts_encodings = self.context_encoder(**contexts)
        logits = torch.matmul(question_encodings, contexts_encodings.T)

        output = {"logits": logits}

        if return_loss and labels is not None:
            if isinstance(self.loss_type, torch.nn.NLLLoss):
                labels = labels.argmax(dim=1)
                logits = F.log_softmax(logits, dim=1)
                if len(question_encodings.size()) > 1:
                    logits = logits.view(question_encodings.size(0), -1)

            output["loss"] = self.loss_type(logits, labels)

        return output
