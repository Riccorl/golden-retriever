from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from data.labels import Labels
import transformers as tr


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        mention_encoder: Union[str, tr.PreTrainedModel] = "bert-base-cased",
        entity_encoder: Union[str, tr.PreTrainedModel] = "bert-base-cased",
        labels: Labels = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if labels is not None:
            self.labels = labels
        if isinstance(mention_encoder, str):
            self.mention_encoder = tr.AutoModel.from_pretrained(mention_encoder)
        else:
            self.mention_encoder = mention_encoder
        if isinstance(entity_encoder, str):
            self.entity_encoder = tr.AutoModel.from_pretrained(entity_encoder)
        else:
            self.entity_encoder = entity_encoder

    def encode(
        self,
        mention_token_ids=None,
        mention_masks=None,
        candidate_token_ids=None,
        candidate_masks=None,
        entity_token_ids=None,
        entity_masks=None,
    ):
        candidates_embeds = None
        mention_embeds = None
        entity_embeds = None
        # candidate_token_ids and mention_token_ids not None during training
        # mention_token_ids not None for embedding mentions during inference
        # entity_token_ids not None for embedding entities during inference
        if candidate_token_ids is not None:
            B, C, L = candidate_token_ids.size()
            candidate_token_ids = candidate_token_ids.view(-1, L)
            candidate_masks = candidate_masks.view(-1, L)
            # B X C X L --> BC X L
            candidates_embeds = self.entity_encoder(
                input_ids=candidate_token_ids, attention_mask=candidate_masks
            )[0][:, 0, :].view(B, C, -1)
        if mention_token_ids is not None:
            mention_embeds = self.mention_encoder(
                input_ids=mention_token_ids, attention_mask=mention_masks
            )[0][:, 0, :]
        if entity_token_ids is not None:
            # for getting all the entity embeddings
            entity_embeds = self.entity_encoder(
                input_ids=entity_token_ids, attention_mask=entity_masks
            )[0][:, 0, :]
        return mention_embeds, candidates_embeds, entity_embeds

    def forward(
        self,
        mention_token_ids=None,
        mention_masks=None,
        candidate_token_ids=None,
        candidate_masks=None,
        passages_labels=None,
        entity_token_ids=None,
        entity_masks=None,
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

        if not self.training:
            return self.encode(
                mention_token_ids,
                mention_masks,
                candidate_token_ids,
                candidate_masks,
                entity_token_ids,
                entity_masks,
            )
        B, C, L = candidate_token_ids.size()
        mention_embeds, candidates_embeds, _ = self.encode(
            mention_token_ids, mention_masks, candidate_token_ids, candidate_masks
        )
        mention_embeds = mention_embeds.unsqueeze(1)
        logits = torch.matmul(mention_embeds, candidates_embeds.transpose(1, 2)).view(
            B, -1
        )
        # loss = self.loss_fct(logits, passages_labels)
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
