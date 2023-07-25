from typing import Any, Dict, Optional, Union

import torch
import transformers as tr
from transformers.activations import GELUActivation

from goldenretriever.common.utils import is_package_available

# check if ORT is available
if is_package_available("onnxruntime"):
    from optimum.onnxruntime import ORTModelForFeatureExtraction


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        language_model: Union[
            str, tr.PreTrainedModel, "ORTModelForFeatureExtraction"
        ] = "sentence-transformers/all-MiniLM-12-v2",
        from_pretrained: bool = True,
        pooling_strategy: str = "mean",
        layer_norm: bool = False,
        layer_norm_eps: float = 1e-12,
        projection_size: Optional[int] = None,
        projection_dropout: float = 0.1,
        load_ort_model: bool = False,
        freeze: bool = False,
        config: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if isinstance(language_model, str):
            if load_ort_model:
                self.language_model = ORTModelForFeatureExtraction.from_pretrained(
                    language_model, from_transformers=True
                )
            else:
                if from_pretrained:
                    self.language_model = tr.AutoModel.from_pretrained(language_model)
                else:
                    self.language_model = tr.AutoModel.from_config(
                        tr.AutoConfig.from_pretrained(language_model)
                    )
        else:
            self.language_model = language_model

        if freeze and not isinstance(self.language_model, ORTModelForFeatureExtraction):
            for param in self.language_model.parameters():
                param.requires_grad = False

        # normalization layer
        self.layer_norm_layer: Optional[torch.nn.LayerNorm] = None
        if layer_norm:
            layer_norm_size = (
                projection_size
                if projection_size is not None
                else self.language_model.config.hidden_size
            )
            self.layer_norm_layer = torch.nn.LayerNorm(
                layer_norm_size, eps=layer_norm_eps
            )

        # projection layer
        self.projection: Optional[torch.nn.Sequential] = None
        if projection_size is not None:
            self.projection = torch.nn.Sequential(
                torch.nn.Dropout(projection_dropout),
                torch.nn.Linear(
                    self.language_model.config.hidden_size,
                    self.language_model.config.hidden_size,
                ),
                GELUActivation(),
                torch.nn.Dropout(projection_dropout),
                torch.nn.Linear(
                    self.language_model.config.hidden_size, projection_size
                ),
            )

        # save the other parameters
        self.language_model_name = self.language_model.config.name_or_path
        self.layer_norm = layer_norm
        self.projection_size = projection_size
        self.projection_dropout = projection_dropout
        self.pooling_strategy = pooling_strategy
        self.load_ort_model = load_ort_model
        self.freeze = freeze

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

        if token_type_ids is not None:
            model_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            model_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)

        model_output = self.language_model(**model_kwargs)
        if pooling_strategy == "cls":
            pooling = model_output.pooler_output
        elif pooling_strategy == "mean":
            # mean pooling
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            mean_pooling = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooling = mean_pooling
        else:
            raise ValueError(
                f"Pooling strategy {pooling_strategy} not supported, use 'cls' or 'mean'"
            )

        if self.projection is not None:
            pooling = self.projection(pooling)

        if self.layer_norm_layer is not None:
            # the normalization layer takes in inout the pooled output and the attention output
            # pooling = pooling + model_output.attentions[-1]
            pooling = self.layer_norm_layer(pooling)

        return pooling

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return the configuration of the model.

        Returns:
            `Dict[str, Any]`: The configuration of the model.
        """
        return dict(
            _target_=f"{self.__class__.__module__}.{self.__class__.__name__}",
            language_model=self.language_model_name,
            layer_norm=self.layer_norm,
            projection_size=self.projection_size,
            projection_dropout=self.projection_dropout,
            pooling_strategy=self.pooling_strategy,
            load_ort_model=self.load_ort_model,
            freeze=self.freeze,
        )
