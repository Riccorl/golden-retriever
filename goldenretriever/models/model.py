import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hydra
import psutil
import torch
import torch.nn.functional as F
import transformers as tr
from omegaconf import OmegaConf
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from goldenretriever.common.log import get_console_logger, get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.common.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    from_cache,
    is_package_available,
    is_remote_url,
    sapienzanlp_model_urls,
)
from goldenretriever.data.base.datasets import BaseDataset
from goldenretriever.data.labels import Labels
from goldenretriever.models.faiss_indexer import FaissIndexer, FaissOutput

# check if ORT is available
if is_package_available("onnxruntime"):
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
    from optimum.onnxruntime.configuration import AutoOptimizationConfig

INDEX_NAME = "index.json"
INDEX_VECTOR_NAME = "index.pt"
FAISS_INDEX_NAME = "faiss_index.bin"

PRECISION_MAP = {
    None: torch.float32,
    16: torch.float16,
    32: torch.float32,
    "float16": torch.float16,
    "float32": torch.float32,
    "half": torch.float16,
    "float": torch.float32,
    "16": torch.float16,
    "32": torch.float32,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

console_logger = get_console_logger()
logger = get_logger(__name__, level=logging.INFO)


@dataclass
class GoldenRetrieverOutput(tr.file_utils.ModelOutput):
    """Class for model's outputs."""

    logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    question_encodings: Optional[torch.FloatTensor] = None
    passages_encodings: Optional[torch.FloatTensor] = None


@dataclass
class RetrievedSample:
    """
    Dataclass for the output of the GoldenRetriever model.
    """

    score: float
    index: int
    label: str


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

        # projection layer
        self.projection: Optional[torch.nn.Sequential] = None
        if projection_size is not None:
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(
                    self.language_model.config.hidden_size, projection_size
                ),
                torch.nn.Dropout(projection_dropout),
            )

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


class GoldenRetriever(torch.nn.Module):
    def __init__(
        self,
        question_encoder: str,
        loss_type: Optional[torch.nn.Module] = None,
        passage_encoder: Optional[str] = None,
        passage_index: Optional[Labels] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.passage_encoder_is_question_encoder = False
        # question encoder model
        # if isinstance(question_encoder, str):
        self.question_encoder = tr.AutoModel.from_pretrained(question_encoder)
        # self.question_encoder = question_encoder
        if not passage_encoder:
            # if no passage encoder is provided,
            # share the weights of the question encoder
            passage_encoder = self.question_encoder
            # keep track of the fact that the passage encoder is the same as the question encoder
            self.passage_encoder_is_question_encoder = True
        else:
            if isinstance(passage_encoder, str):
                passage_encoder = tr.AutoModel.from_pretrained(passage_encoder)
        # passage encoder model
        self.passage_encoder = passage_encoder

        # dropout
        self.question_dropout = torch.nn.Dropout(kwargs.get("dropout", 0.1))
        self.passage_dropout = torch.nn.Dropout(kwargs.get("dropout", 0.1))

        # loss function
        self.loss_type = loss_type

        # indexer stuff, lazy loaded
        self._passage_index: Optional[Labels] = passage_index
        self._passage_embeddings: Optional[torch.Tensor] = None
        self._faiss_indexer: Optional[FaissIndexer] = None

        # lazy load the tokenizer for inference
        self._question_tokenizer: Optional[tr.PreTrainedTokenizer] = None
        self._passage_tokenizer: Optional[tr.PreTrainedTokenizer] = None

    @staticmethod
    def encoder_forward(
        encoder: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if token_type_ids is not None:
            model_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            model_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_states = encoder(**model_kwargs).last_hidden_state
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(
        self,
        questions: Optional[Dict[str, torch.Tensor]] = None,
        passages: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        question_encodings: Optional[torch.Tensor] = None,
        passages_encodings: Optional[torch.Tensor] = None,
        passages_per_question: Optional[List[int]] = None,
        return_loss: bool = False,
        return_encodings: bool = False,
        *args,
        **kwargs,
    ) -> GoldenRetrieverOutput:
        """
        Forward pass of the model.

        Args:
            questions (`Dict[str, torch.Tensor]`):
                The questions to encode.
            passages (`Dict[str, torch.Tensor]`):
                The passages to encode.
            labels (`torch.Tensor`):
                The labels of the sentences.
            return_loss (`bool`):
                Whether to compute the predictions.
            question_encodings (`torch.Tensor`):
                The encodings of the questions.
            passages_encodings (`torch.Tensor`):
                The encodings of the passages.
            passages_per_question (`List[int]`):
                The number of passages per question.
            return_loss (`bool`):
                Whether to compute the loss.
            return_encodings (`bool`):
                Whether to return the encodings.

        Returns:
            obj:`torch.Tensor`: The outputs of the model.
        """
        if questions is None and question_encodings is None:
            raise ValueError(
                "Either `questions` or `question_encodings` must be provided"
            )
        if passages is None and passages_encodings is None:
            raise ValueError(
                "Either `passages` or `passages_encodings` must be provided"
            )

        if question_encodings is None:
            # print(questions)
            question_encodings = self.encoder_forward(
                **{**{"encoder": self.question_encoder}, **questions}
            )
            question_encodings = self.question_dropout(question_encodings)
        if passages_encodings is None:
            passages_encodings = self.encoder_forward(
                **{**{"encoder": self.passage_encoder}, **passages}
            )
            passages_encodings = self.passage_dropout(passages_encodings)

        if passages_per_question is not None:
            # multiply each question encoding with a passages_per_question encodings
            concatenated_passages = torch.stack(
                torch.split(passages_encodings, passages_per_question)
            ).transpose(1, 2)
            if isinstance(self.loss_type, torch.nn.BCEWithLogitsLoss):
                # normalize the encodings for cosine similarity
                concatenated_passages = F.normalize(concatenated_passages, p=2, dim=2)
                question_encodings = F.normalize(question_encodings, p=2, dim=1)
            logits = torch.bmm(
                question_encodings.unsqueeze(1), concatenated_passages
            ).view(question_encodings.shape[0], -1)
        else:
            if isinstance(self.loss_type, torch.nn.BCEWithLogitsLoss):
                # normalize the encodings for cosine similarity
                question_encodings = F.normalize(question_encodings, p=2, dim=1)
                passages_encodings = F.normalize(passages_encodings, p=2, dim=1)

            logits = torch.matmul(question_encodings, passages_encodings.T)

        output = dict(logits=logits)

        if return_loss and labels is not None:
            if self.loss_type is None:
                raise ValueError(
                    "If `return_loss` is set to `True`, `loss_type` must be provided"
                )
            if isinstance(self.loss_type, torch.nn.NLLLoss):
                labels = labels.argmax(dim=1)
                logits = F.log_softmax(logits, dim=1)
                if len(question_encodings.size()) > 1:
                    logits = logits.view(question_encodings.size(0), -1)

            output["loss"] = self.loss_type(logits, labels)

        if return_encodings:
            output["question_encodings"] = question_encodings
            output["passages_encodings"] = passages_encodings

        return GoldenRetrieverOutput(**output)

    @torch.no_grad()
    @torch.inference_mode()
    def index(
        self,
        passages: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        passage_max_length: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        force_reindex: bool = False,
        use_faiss: bool = False,
        use_ort: bool = False,
        move_index_to_cpu: bool = False,
        precision: Optional[Union[str, int]] = None,
        index_precision: Optional[Union[str, int]] = None,
    ):
        """
        Index the passages for later retrieval.

        Args:
            passages (`List[str]`):
                The passages to index.
            batch_size (`int`):
                The batch size to use for the indexing.
            num_workers (`int`):
                The number of workers to use for the indexing.
            passage_max_length (`Optional[int]`):
                The maximum length of the passages.
            collate_fn (`Callable`):
                The collate function to use for the indexing.
            force_reindex (`bool`):
                Whether to force reindexing even if the passages are already indexed.
            use_faiss (`bool`):
                Whether to use faiss for the indexing.
            use_ort (`bool`):
                Whether to use onnxruntime for the indexing.
            move_index_to_cpu (`bool`):
                Whether to move the index to the CPU after the indexing.
            precision (`Optional[Union[str, int]]`):
                The precision to use for the model.
            index_precision (`Optional[Union[str, int]]`):
                The precision to use for the index.
        """
        if self._passage_embeddings is not None and not force_reindex:
            return

        if self._faiss_indexer is not None and not force_reindex and use_faiss:
            return

        # release the memory
        if collate_fn is None:
            tokenizer = self.passage_tokenizer
            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=passage_max_length or tokenizer.model_max_length,
                )
            )
        dataloader = DataLoader(
            BaseDataset(name="passage", data=passages),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        # we can use the onnx runtime optimized encoder for the indexing
        if not use_ort:
            passage_encoder = self.passage_encoder
        else:
            passage_encoder = self._load_ort_optimized_encoder(self.passage_encoder)
        # Create empty lists to store the passage embeddings and passage index
        passage_embeddings: List[torch.Tensor] = []

        # fucking autocast only wants pure strings like 'cpu' or 'cuda'
        # we need to convert the model device to that
        device_type_for_autocast = str(self.device).split(":")[0]
        # autocast doesn't work with CPU and stuff different from bfloat16
        autocast_pssg_mngr = (
            contextlib.nullpassage()
            if device_type_for_autocast == "cpu"
            else (
                torch.autocast(
                    device_type=device_type_for_autocast,
                    dtype=PRECISION_MAP[precision],
                )
            )
        )
        with autocast_pssg_mngr:
            # Iterate through each batch in the dataloader
            for batch in tqdm(dataloader, desc="Indexing"):
                # Move the batch to the device
                batch: ModelInputs = batch.to(self.device)
                # Compute the passage embeddings
                passage_outs = self.encoder_forward(**{**{"encoder": passage_encoder}, **batch})
                # Append the passage embeddings to the list
                if move_index_to_cpu:
                    passage_embeddings.extend([c.detach().cpu() for c in passage_outs])
                else:
                    passage_embeddings.extend([c for c in passage_outs])

        # move the passage embeddings to the CPU if not already done
        # the move to cpu and then to gpu is needed to avoid OOM when using mixed precision
        if not move_index_to_cpu:
            passage_embeddings = [c.detach().cpu() for c in passage_embeddings]
        # stack it
        passage_embeddings: torch.Tensor = torch.stack(passage_embeddings, dim=0)
        # move the passage embeddings to the gpu if needed
        if not move_index_to_cpu:
            if index_precision:
                passage_embeddings = passage_embeddings.to(
                    PRECISION_MAP[index_precision]
                )
            passage_embeddings = passage_embeddings.to(self.device)
        self._passage_embeddings = passage_embeddings

        # free up memory from the unused variable
        del passage_embeddings

        # Create a dictionary mapping the passage index to the passage
        self._passage_index = Labels()
        self._passage_index.add_labels(
            {passage: i for i, passage in enumerate(passages)}
        )
        if use_faiss and (self._faiss_indexer is None or force_reindex):
            self._faiss_indexer = FaissIndexer(
                embeddings=self._passage_embeddings, use_gpu=bool(not move_index_to_cpu)
            )
            # free up memory
            self._passage_embeddings = None

    @torch.no_grad()
    @torch.inference_mode()
    def retrieve(
        self,
        text: Optional[Union[str, List[str]]] = None,
        text_pair: Optional[Union[str, List[str]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
        max_length: Optional[int] = None,
        precision: Optional[Union[str, int]] = None,
    ) -> List[List[RetrievedSample]]:
        """
        Retrieve the passages for the questions.

        Args:
            text (`Optional[Union[str, List[str]]]`):
                The questions to retrieve the passages for.
            text_pair (`Optional[Union[str, List[str]]]`):
                The questions to retrieve the passages for.
            input_ids (`torch.Tensor`):
                The input ids of the questions.
            attention_mask (`torch.Tensor`):
                The attention mask of the questions.
            token_type_ids (`torch.Tensor`):
                The token type ids of the questions.
            k (`int`):
                The number of top passages to retrieve.
            max_length (`Optional[int]`):
                The maximum length of the questions.
            precision (`Optional[Union[str, int]]`):
                The precision to use for the model.

        Returns:
            `List[List[RetrievedSample]]`: The retrieved passages and their indices.
        """
        if self._passage_embeddings is None and self._faiss_indexer is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        if text is None and input_ids is None:
            raise ValueError(
                "Either `text` or `input_ids` must be provided to retrieve the passages."
            )

        if k is None:
            k = self._passage_embeddings.size(0)

        if text is not None:
            if isinstance(text, str):
                text = [text]
            if text_pair is not None and isinstance(text_pair, str):
                text_pair = [text_pair]
            tokenizer = self.question_tokenizer
            model_inputs = ModelInputs(
                tokenizer(
                    text,
                    text_pair=text_pair,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length or tokenizer.model_max_length,
                )
            )
        else:
            model_inputs = ModelInputs(dict(input_ids=input_ids))
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids

        model_inputs.to(self.device)

        if self._faiss_indexer is not None:
            faiss_outs: FaissOutput = self._faiss_indexer.search(
                self.question_encoder(**model_inputs), k=k
            )
            batch_top_k: torch.Tensor = faiss_outs.indices
            batch_scores: torch.Tensor = faiss_outs.distances
        else:
            # fucking autocast only wants pure strings like 'cpu' or 'cuda'
            # we need to convert the model device to that
            device_type_for_autocast = str(self.device).split(":")[0]
            # autocast doesn't work with CPU and stuff different from bfloat16
            autocast_pssg_mngr = (
                contextlib.nullpassage()
                if device_type_for_autocast == "cpu"
                else (
                    torch.autocast(
                        device_type=device_type_for_autocast,
                        dtype=PRECISION_MAP[precision],
                    )
                )
            )
            with autocast_pssg_mngr:
                # check if the device of the passage embeddings is the same
                # as the device of the input ids
                if self._passage_embeddings.device != model_inputs.input_ids.device:
                    self._passage_embeddings = self._passage_embeddings.to(
                        model_inputs.input_ids.device
                    )
                model_inputs = dict(
                    questions=model_inputs,
                    passages_encodings=self._passage_embeddings,
                )
                # check that the device of the question encoder is the same as
                # the device of the passage embeddings
                past_device = self.device
                if past_device != self._passage_embeddings:
                    self.to(self._passage_embeddings.device)
                # Compute the similarity between the questions and the passages
                similarity = self(**model_inputs)["logits"]
                # move the model back to the original device
                if past_device != self.device:
                    self.to(past_device)
                # Retrieve the indices of the top k passage embeddings
                retriever_out: Tuple = torch.topk(
                    similarity, k=min(k, similarity.shape[-1]), dim=1
                )
                batch_top_k: torch.Tensor = retriever_out.indices
                batch_scores: torch.Tensor = retriever_out.values
        # get int values
        batch_top_k: List[List[int]] = batch_top_k.detach().cpu().tolist()
        # get float values
        batch_scores: List[List[float]] = batch_scores.detach().cpu().tolist()
        # Retrieve the passages corresponding to the indices
        batch_passages = [
            [self._passage_index.get_label_from_index(i) for i in indices]
            for indices in batch_top_k
        ]
        # build the output object
        batch_retrieved_samples = [
            [
                RetrievedSample(label=passage, index=index, score=score)
                for passage, index, score in zip(passages, indices, scores)
            ]
            for passages, indices, scores in zip(
                batch_passages, batch_top_k, batch_scores
            )
        ]
        # return
        return batch_retrieved_samples

    def get_index_from_passage(self, passage: str) -> int:
        """
        Get the index of the passage.

        Args:
            passage (`str`):
                The passage to get the index for.

        Returns:
            `int`: The index of the passage.
        """
        if self._passage_embeddings is None and self._faiss_indexer is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self._passage_index.get_index_from_label(passage)

    def get_passage_from_index(self, index: int) -> str:
        """
        Get the passage from the index.

        Args:
            index (`int`):
                The index of the passage.

        Returns:
            `str`: The passage.
        """
        if self._passage_embeddings is None and self._faiss_indexer is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self._passage_index.get_label_from_index(index)

    def get_vector_from_index(self, index: int) -> torch.Tensor:
        """
        Get the passage vector from the index.

        Args:
            index (`int`):
                The index of the passage.

        Returns:
            `torch.Tensor`: The passage vector.
        """
        if self._passage_embeddings is None and self._faiss_indexer is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        if self._passage_embeddings is None:
            return self._faiss_indexer.reconstruct(index)
        return self._passage_embeddings[index]

    def get_vector_from_passage(self, passage: str) -> torch.Tensor:
        """
        Get the passage vector from the passage.

        Args:
            passage (`str`):
                The passage.

        Returns:
            `torch.Tensor`: The passage vector.
        """
        if self._passage_embeddings is None and self._faiss_indexer is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.get_vector_from_index(self.get_index_from_passage(passage))

    @property
    def question_tokenizer(self) -> tr.PreTrainedTokenizer:
        """
        The question tokenizer.
        """
        if self._question_tokenizer:
            return self._question_tokenizer

        if (
            self.question_encoder.name_or_path
            == self.question_encoder.name_or_path
        ):
            if not self._question_tokenizer:
                self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                    self.question_encoder.name_or_path
                )
            self._passage_tokenizer = self._question_tokenizer
            return self._question_tokenizer

        if not self._question_tokenizer:
            self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                self.question_encoder.name_or_path
            )
        return self._question_tokenizer

    @property
    def passage_tokenizer(self) -> tr.PreTrainedTokenizer:
        """
        The passage tokenizer.
        """
        if self._passage_tokenizer:
            return self._passage_tokenizer

        if (
            self.question_encoder.language_model_name
            == self.passage_encoder.language_model_name
        ):
            if not self._question_tokenizer:
                self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                    self.question_encoder.language_model_name
                )
            self._passage_tokenizer = self._question_tokenizer
            return self._passage_tokenizer

        if not self._passage_tokenizer:
            self._passage_tokenizer = tr.AutoTokenizer.from_pretrained(
                self.passage_encoder.language_model_name
            )
        return self._passage_tokenizer

    @property
    def passage_embeddings(self) -> torch.Tensor:
        """
        The passage embeddings.
        """
        return self._passage_embeddings

    @property
    def passage_index(self) -> Labels:
        """
        The passage index.
        """
        return self._passage_index

    @property
    def device(self) -> torch.device:
        """
        The device of the model.
        """
        return next(self.parameters()).device

    @staticmethod
    def _load_ort_optimized_encoder(
        encoder: SentenceEncoder, provider: str = "CPUExecutionProvider"
    ) -> SentenceEncoder:
        """
        Load an optimized ONNX Runtime encoder.

        Args:
            encoder (`SentenceEncoder`):
                The encoder to optimize.
            provider (`str`, optional):
                The ONNX Runtime provider to use. Defaults to "CPUExecutionProvider".

        Returns:
            `SentenceEncoder`: The optimized encoder.
        """

        temp_dir = tempfile.mkdtemp()
        encoder.language_model.save_pretrained(temp_dir)
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            temp_dir, export=True, provider=provider, use_io_binding=True
        )
        optimizer = ORTOptimizer.from_pretrained(ort_model)
        optimization_config = AutoOptimizationConfig.O4()
        optimizer.optimize(save_dir=temp_dir, optimization_config=optimization_config)
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            temp_dir,
            export=True,
            provider=provider,
            use_io_binding=bool(provider == "CUDAExecutionProvider"),
        )
        return SentenceEncoder(
            language_model=ort_model,
            pooling_strategy=encoder.pooling_strategy,
        )

    @property
    def config(self) -> Dict[str, Any]:
        """
        The configuration of the retriever.

        Returns:
            `Dict[str, Any]`: The configuration of the retriever.
        """
        return dict(
            _target_=f"{self.__class__.__module__}.{self.__class__.__name__}",
            question_encoder=self.question_encoder.name_or_path,
            passage_encoder=self.passage_encoder.name_or_path
            if not self.passage_encoder_is_question_encoder
            else None,
            loss_type=dict(
                _target_=f"{self.loss_type.__class__.__module__}.{self.loss_type.__class__.__name__}"
            ),
            # passage_index is not saved because it might be too large
            passage_index=None,
        )

    def save_pretrained(
        self,
        output_dir: Union[str, os.PathLike],
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ):
        """
        Save the retriever to a directory.

        Args:
            output_dir (`str`):
                The directory to save the retriever to.
            config (`Optional[Dict[str, Any]]`, `optional`):
                The configuration to save. If `None`, the current configuration of the retriever will be
                saved. Defaults to `None`.
        """

        if config is None:
            # create a default config
            config = self.config

        # create the output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving retriever to {output_dir}")
        logger.info(f"Saving config to {output_dir / CONFIG_NAME}")
        # pretty print the config
        pprint(config, console=console_logger, expand_all=True)

        # override the from_pretrained parameter of the encoders
        # we don't want to load the pretrained weights from HF Hub when loading the retriever
        # config["question_encoder"]["from_pretrained"] = False
        # if config["passage_encoder"] is not None:
        #     config["passage_encoder"]["from_pretrained"] = False
        # save the config using OmegaConf
        OmegaConf.save(config, output_dir / CONFIG_NAME)

        if self._passage_embeddings is None or self._passage_index is None:
            raise ValueError("The passages must be indexed before they can be saved.")

        if not self.passage_encoder_is_question_encoder:
            self.question_encoder.save_pretrained(output_dir / model_name + "_question")
            self.question_tokenizer.save_pretrained(
                output_dir / model_name + "_question"
            )
            self.passage_encoder.save_pretrained(output_dir / model_name + "_passage")
            self.passage_tokenizer.save_pretrained(output_dir / model_name + "_passage")
        else:
            self.question_encoder.save_pretrained(output_dir / model_name)
            self.question_tokenizer.save_pretrained(output_dir / model_name)

        # save the current state of the retriever
        logger.info(f"Saving retriever state to {output_dir / WEIGHTS_NAME}")
        torch.save(self.state_dict(), output_dir / WEIGHTS_NAME)
        # save the passage embeddings
        logger.info(f"Saving passage embeddings to {output_dir / INDEX_VECTOR_NAME}")
        torch.save(self._passage_embeddings, output_dir / INDEX_VECTOR_NAME)
        # save the passage index
        logger.info(f"Saving passage index to {output_dir / INDEX_NAME}")
        self._passage_index.save(output_dir / INDEX_NAME)
        # save the faiss indexer
        # if self._faiss_indexer is not None:
        #     logger.log(f"Saving faiss index to {output_dir / FAISS_INDEX_NAME}")
        #     self._faiss_indexer.save(output_dir)

        logger.info("Saving retriever to disk done.")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_dir: Union[str, Path],
        strict: bool = True,
        device: str = "cpu",
        index_device: str = "cpu",
        index_precision: Optional[str] = None,
        load_faiss_index: bool = False,
        load_index_vector: bool = True,
        compile: bool = False,
        filenames: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> "GoldenRetriever":
        """
        Load a retriever from disk.

        Args:
            model_name_or_dir (`str` or `Path`):
                The path to the directory containing the retriever files.
            strict (`bool`, optional):
                Whether to raise an error if the state dict of the saved retriever does not
                match the state dict of the current retriever. Defaults to `True`.
            device (`str`, optional):
                The device to load the retriever to. Defaults to `cpu`.
            index_device (`str`, optional):
                The device to load the index to. Defaults to `cpu`.
            index_precision (`str`, optional):
                The precision to load the index to. Defaults to None.
            load_index_vector   (`bool`, optional):
                Whether to load the index vector. Defaults to `True`.
            load_faiss_index (`bool`, optional):
                Whether to load the faiss index. Defaults to `False`.
            compile (`bool`, optional):
                Whether to compile the model using Torch 2.0. Defaults to `False`.
            filenames (`Optional[List[str]]`, optional):
                The filenames of the files to load. If `None`, the default filenames will be used.
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        Returns:
            `GoldenRetriever`: The retriever.
        """

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)

        if is_remote_url(model_name_or_dir):
            # if model_name_or_dir is a URL
            # download it and try to load
            model_archive = model_name_or_dir
        elif Path(model_name_or_dir).is_dir() or Path(model_name_or_dir).is_file():
            # if model_name_or_dir is a local directory or
            # an archive file try to load it
            model_archive = model_name_or_dir
        else:
            # probably model_name_or_dir is a sapienzanlp model id
            # guess the url and try to download
            model_name_or_dir_ = model_name_or_dir
            # raise ValueError(f"Providing a model id is not supported yet.")
            model_archive = sapienzanlp_model_urls(model_name_or_dir_)

        if filenames is None:
            filenames = [CONFIG_NAME, WEIGHTS_NAME, INDEX_VECTOR_NAME, INDEX_NAME]

        model_dir = from_cache(
            model_archive,
            filenames=filenames,
            cache_dir=cache_dir,
            force_download=force_download,
        )

        logger.info(f"Loading retriever from {model_dir}")
        # get model stuff
        if device == "cpu":
            num_threads = os.getenv(
                "TORCH_NUM_THREADS", psutil.cpu_count(logical=False)
            )
            torch.set_num_threads(num_threads)
            logger.info(f"Model is running on {num_threads} threads")

        # parse config file
        config_path = model_dir / CONFIG_NAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found at {config_path}."
            )
        config = OmegaConf.load(config_path)
        pprint(OmegaConf.to_container(config), console=console_logger, expand_all=True)

        # load the index vocabulary
        passage_index = Labels.from_file(model_dir / INDEX_NAME)

        weights_path = model_dir / WEIGHTS_NAME

        # load model from config
        logger.info("Loading model")
        model = hydra.utils.instantiate(
            config, passage_index=passage_index, *args, **kwargs
        )
        # load model weights
        model_state = torch.load(weights_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(
            model_state, strict=strict
        )
        if unexpected_keys or missing_keys:
            logger.info(
                f"Error loading state dict for {model.__class__.__name__}\n\t"
                f"Missing keys: {missing_keys}\n\t"
                f"Unexpected keys: {unexpected_keys}"
            )

        if load_index_vector:
            # run some checks
            index_vectors = model_dir / INDEX_VECTOR_NAME
            faiss_index_vectors = model_dir / FAISS_INDEX_NAME
            if load_faiss_index and not faiss_index_vectors.exists():
                logger.info(
                    f"{FAISS_INDEX_NAME} does not exist. Trying to convert from dense index."
                )
            if not index_vectors.exists():
                raise ValueError(f"Index vectors `{index_vectors}` does not exist.")
            if not (model_dir / INDEX_NAME).exists():
                raise ValueError(f"Index `{model_dir / INDEX_NAME}` does not exist.")

            # select between faiss and dense index
            logger.info("Loading index vectors")
            embeddings = torch.load(index_vectors, map_location="cpu")
            if index_precision and embeddings.dtype != PRECISION_MAP[index_precision]:
                logger.info(
                    f"Index vectors are of type {embeddings.dtype}. "
                    f"Converting to {PRECISION_MAP[index_precision]}."
                )
                embeddings = embeddings.to(PRECISION_MAP[index_precision])

            if load_faiss_index:
                faiss_kwargs = {
                    "use_gpu": bool(index_device == "cuda"),
                }
                if not faiss_index_vectors.exists():
                    faiss_kwargs.update({"embeddings": embeddings})
                    model._faiss_indexer = FaissIndexer(**faiss_kwargs)
                    del embeddings
                else:
                    faiss_kwargs.update({"loading_dir": model_dir})
                    model._faiss_indexer = FaissIndexer.load(**faiss_kwargs)
            else:
                if device == "cuda":
                    embeddings = embeddings.to(device)
                model._passage_embeddings = embeddings

        # move model to device
        model.to(device)

        if compile:
            try:
                model = torch.compile(model)
            except Exception as e:
                # show the error message
                print(e)
                logger.info(
                    f"Failed to compile the model, you may need to install PyTorch 2.x"
                )

        return model
