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

from golden_retriever.common.log import get_console_logger
from golden_retriever.common.model_inputs import ModelInputs
from golden_retriever.common.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    from_cache,
    is_package_available,
    is_remote_url,
)
from golden_retriever.data.datasets import BaseDataset
from golden_retriever.data.labels import Labels
from golden_retriever.models.faiss_indexer import FaissIndexer, FaissOutput

# check if ORT is available
if is_package_available("onnxruntime"):
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
    from optimum.onnxruntime.configuration import AutoOptimizationConfig

INDEX_NAME = "index.json"
INDEX_VECTOR_NAME = "index.pt"
FAISS_INDEX_NAME = "faiss_index.bin"

PRECISION_MAP = {
    None: None,
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

logger = get_console_logger()


@dataclass
class GoldenRetrieverOutput(tr.file_utils.ModelOutput):
    """Class for model's outputs."""

    logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    question_encodings: Optional[torch.FloatTensor] = None
    contexts_encodings: Optional[torch.FloatTensor] = None


@dataclass
class RetrieveOutput:
    """
    Dataclass for the output of the GoldenRetriever model.
    """

    scores: List[List[float]]
    indices: List[List[int]]
    contexts: List[List[str]]


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        language_model: Union[
            str, tr.PreTrainedModel, "ORTModelForFeatureExtraction"
        ] = "sentence-transformers/all-MiniLM-12-v2",
        from_pretrained: bool = True,
        pooling_strategy: str = "mean",
        layer_norm: bool = False,
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
            self.layer_norm_layer = torch.nn.LayerNorm(
                self.language_model.config.hidden_size
            )

        # projection layer
        self.projection: Optional[torch.nn.Sequential] = None
        if projection_size is not None:
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(
                    self.language_model.config.hidden_size, projection_size
                ),
                torch.nn.Dropout(projection_dropout),
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

        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

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

        if self.layer_norm_layer is not None:
            pooling = self.layer_norm_layer(pooling)

        if self.projection is not None:
            pooling = self.projection(pooling)

        return pooling

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return the configuration of the model.

        Returns:
            `Dict[str, Any]`: The configuration of the model.
        """
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "language_model": self.language_model_name,
            "layer_norm": self.layer_norm,
            "projection_size": self.projection_size,
            "projection_dropout": self.projection_dropout,
            "pooling_strategy": self.pooling_strategy,
            "load_ort_model": self.load_ort_model,
            "freeze": self.freeze,
        }


class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class GoldenRetriever(torch.nn.Module):
    def __init__(
        self,
        question_encoder: SentenceEncoder,
        loss_type: Optional[torch.nn.Module] = None,
        context_encoder: Optional[SentenceEncoder] = None,
        context_index: Optional[Labels] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.context_encoder_is_question_encoder = False
        # question encoder model
        self.question_encoder = question_encoder
        if not context_encoder:
            # if no context encoder is provided,
            # share the weights of the question encoder
            context_encoder = question_encoder
            # keep track of the fact that the context encoder is the same as the question encoder
            self.context_encoder_is_question_encoder = True
        # context encoder model
        self.context_encoder = context_encoder

        # loss function
        self.loss_type = loss_type

        # indexer stuff, lazy loaded
        self._context_index: Optional[Labels] = context_index
        self._context_embeddings: Optional[torch.Tensor] = None
        self._faiss_indexer: Optional[FaissIndexer] = None

        # lazy load the tokenizer for inference
        self._question_tokenizer: Optional[tr.PreTrainedTokenizer] = None
        self._context_tokenizer: Optional[tr.PreTrainedTokenizer] = None

    def forward(
        self,
        questions: Optional[Dict[str, torch.Tensor]] = None,
        contexts: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        question_encodings: Optional[torch.Tensor] = None,
        contexts_encodings: Optional[torch.Tensor] = None,
        contexts_per_question: Optional[List[int]] = None,
        inner_batch_size: Optional[int] = None,
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
            contexts_per_question (`List[int]`):
                The number of contexts per question.
            inner_batch_size (`int`):
                The batch size to use for the context encoding.
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
        if contexts is None and contexts_encodings is None:
            raise ValueError(
                "Either `contexts` or `contexts_encodings` must be provided"
            )

        if question_encodings is None:
            question_encodings = self.question_encoder(**questions)
        if contexts_encodings is None:
            contexts_encodings = self.context_encoder(**contexts)

        if contexts_per_question is not None:
            # multiply each question encoding with a contexts_per_question encodings
            concatenated_contexts = torch.stack(
                torch.split(contexts_encodings, contexts_per_question)
            ).transpose(1, 2)
            if isinstance(self.loss_type, torch.nn.BCEWithLogitsLoss):
                # normalize the encodings for cosine similarity
                concatenated_contexts = F.normalize(concatenated_contexts, p=2, dim=2)
                question_encodings = F.normalize(question_encodings, p=2, dim=1)
            logits = torch.bmm(
                question_encodings.unsqueeze(1), concatenated_contexts
            ).view(question_encodings.shape[0], -1)
        else:
            if isinstance(self.loss_type, torch.nn.BCEWithLogitsLoss):
                # normalize the encodings for cosine similarity
                question_encodings = F.normalize(question_encodings, p=2, dim=1)
                contexts_encodings = F.normalize(contexts_encodings, p=2, dim=1)

            # if inner_batch_size is not None:
            #     # split the contexts into batches
            #     contexts_encodings = torch.split(contexts_encodings, 10_000)
            #     logits = []
            #     for batch in contexts_encodings:
            #         # logits.append(torch.einsum("ij,ij->i", question_encodings, batch))
            #         logits.append(torch.matmul(question_encodings, batch.T))
            #     logits = torch.cat(logits)
            # else:
            logits = torch.matmul(question_encodings, contexts_encodings.T)

        output = {"logits": logits}

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
            output["contexts_encodings"] = contexts_encodings

        return GoldenRetrieverOutput(**output)

    @torch.no_grad()
    def index(
        self,
        contexts: List[str],
        batch_size: int = 32,
        num_workers: int = 0,
        context_max_length: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        force_reindex: bool = False,
        use_faiss: bool = False,
        use_ort: bool = False,
        move_index_to_cpu: bool = False,
        precision: Optional[str] = None,
        index_precision: Optional[str] = None,
    ):
        """
        Index the contexts for later retrieval.

        Args:
            contexts (`List[str]`):
                The contexts to index.
            batch_size (`int`):
                The batch size to use for the indexing.
            num_workers (`int`):
                The number of workers to use for the indexing.
            context_max_length (`Optional[int]`):
                The maximum length of the contexts.
            collate_fn (`Callable`):
                The collate function to use for the indexing.
            force_reindex (`bool`):
                Whether to force reindexing even if the contexts are already indexed.
            use_faiss (`bool`):
                Whether to use faiss for the indexing.
            use_ort (`bool`):
                Whether to use onnxruntime for the indexing.
            move_index_to_cpu (`bool`):
                Whether to move the index to the CPU after the indexing.
        """
        if self._context_embeddings is not None and not force_reindex:
            return

        if self._faiss_indexer is not None and not force_reindex and use_faiss:
            return

        if collate_fn is None:
            tokenizer = self.context_tokenizer
            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=context_max_length or tokenizer.model_max_length,
                )
            )
        dataloader = DataLoader(
            BaseDataset(name="context", data=contexts),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        # we can use the onnx runtime optimized encoder for the indexing
        if not use_ort:
            context_encoder = self.context_encoder
        else:
            context_encoder = self._load_ort_optimized_encoder(self.context_encoder)
        # Create empty lists to store the context embeddings and context index
        context_embeddings: List[torch.Tensor] = []

        # fucking autocast only wants pure strings like 'cpu' or 'cuda'
        # we need to convert the model device to that
        device_type_for_autocast = str(self.device).split(":")[0]
        with torch.autocast(
            device_type=device_type_for_autocast,
            dtype=PRECISION_MAP[precision],
        ):
            # Iterate through each batch in the dataloader
            for batch in tqdm(dataloader, desc="Indexing"):
                # Move the batch to the device
                batch: ModelInputs = batch.to(self.device)
                # Compute the context embeddings
                context_outs = context_encoder(**batch)
                # Append the context embeddings to the list
                context_embeddings.extend([c for c in context_outs])

        # move the context embeddings to the CPU if required
        if move_index_to_cpu:
            context_embeddings = [c.detach().cpu() for c in context_embeddings]

        # if (
        #     index_precision is not None
        #     and not use_faiss
        #     and not move_index_to_cpu
        #     and self.device.type != "cpu"
        # ):
        #     # move it to cpu first
        #     context_embeddings = [c.detach().cpu() for c in context_embeddings]
        #     # then cast it to the desired precision
        #     context_embeddings = [
        #         c.to(PRECISION_MAP[index_precision]) for c in context_embeddings
        #     ]
        #     # then move it back to the device
        #     context_embeddings = [c.to(self.device) for c in context_embeddings]

        # Stack the context embeddings into a tensor and return it along with the context index
        self._context_embeddings = None
        self._context_embeddings = torch.stack(context_embeddings, dim=0)
        # free up memory
        del context_embeddings
        # Create a dictionary mapping the context index to the context
        self._context_index = Labels()
        self._context_index.add_labels(
            {context: i for i, context in enumerate(contexts)}
        )
        if use_faiss and (self._faiss_indexer is None or force_reindex):
            self._faiss_indexer = FaissIndexer(
                embeddings=self._context_embeddings, use_gpu=bool(not move_index_to_cpu)
            )
            # free up memory
            self._context_embeddings = None

    def retrieve(
        self,
        text: Optional[Union[str, List[str]]] = None,
        text_pair: Optional[Union[str, List[str]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
        max_length: Optional[int] = None,
        precision: Optional[str] = None,
    ) -> RetrieveOutput:
        """
        Retrieve the contexts for the questions.

        Args:
            text (`Optional[Union[str, List[str]]]`):
                The questions to retrieve the contexts for.
            text_pair (`Optional[Union[str, List[str]]]`):
                The questions to retrieve the contexts for.
            input_ids (`torch.Tensor`):
                The input ids of the questions.
            attention_mask (`torch.Tensor`):
                The attention mask of the questions.
            token_type_ids (`torch.Tensor`):
                The token type ids of the questions.
            k (`int`):
                The number of top contexts to retrieve.
            max_length (`Optional[int]`):
                The maximum length of the questions.

        Returns:
            `Tuple[List[List[str]], List[List[int]]]`: The retrieved contexts and their indices.
        """
        if self._context_embeddings is None and self._faiss_indexer is None:
            raise ValueError(
                "The contexts must be indexed before they can be retrieved."
            )
        if text is None and input_ids is None:
            raise ValueError(
                "Either `text` or `input_ids` must be provided to retrieve the contexts."
            )

        if k is None:
            k = self._context_embeddings.size(0)

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
            model_inputs.to(self.device)
        else:
            model_inputs = ModelInputs({"input_ids": input_ids})
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids
        if self._faiss_indexer is not None:
            faiss_outs: FaissOutput = self._faiss_indexer.search(
                self.question_encoder(**model_inputs), k=k
            )
            top_k: torch.Tensor = faiss_outs.indices
            scores: torch.Tensor = faiss_outs.distances
        else:
            # fucking autocast only wants pure strings like 'cpu' or 'cuda'
            # we need to convert the model device to that
            device_type_for_autocast = str(self.device).split(":")[0]
            with torch.autocast(
                device_type=device_type_for_autocast,
                dtype=PRECISION_MAP[precision],
            ):
                # check if the device of the context embeddings is the same
                # as the device of the input ids
                if self._context_embeddings.device != model_inputs.input_ids.device:
                    self._context_embeddings = self._context_embeddings.to(
                        model_inputs.input_ids.device
                    )
                model_inputs = {
                    "questions": model_inputs,
                    "contexts_encodings": self._context_embeddings,
                }
                # check that the device of the question encoder is the same as
                # the device of the context embeddings
                past_device = self.device
                if past_device != self._context_embeddings:
                    self.to(self._context_embeddings.device)
                # Compute the similarity between the questions and the contexts
                similarity = self(**model_inputs)["logits"]
                # move the model back to the original device
                if past_device != self.device:
                    self.to(past_device)
                # Retrieve the indices of the top k context embeddings
                retriever_out: Tuple = torch.topk(
                    similarity, k=min(k, similarity.shape[-1]), dim=1
                )
                top_k: torch.Tensor = retriever_out.indices
                scores: torch.Tensor = retriever_out.values
        # get int values
        top_k: List[List[int]] = top_k.cpu().tolist()
        # get float values
        scores: List[List[float]] = scores.cpu().tolist()
        # Retrieve the contexts corresponding to the indices
        contexts = [
            [self._context_index.get_label_from_index(i) for i in indices]
            for indices in top_k
        ]
        # return contexts, top_k
        return RetrieveOutput(contexts=contexts, indices=top_k, scores=scores)

    def get_index_from_context(self, context: str) -> int:
        """
        Get the index of the context.

        Args:
            context (`str`):
                The context to get the index for.

        Returns:
            `int`: The index of the context.
        """
        if self._context_embeddings is None:
            raise ValueError(
                "The contexts must be indexed before they can be retrieved."
            )
        return self._context_index.get_index_from_label(context)

    def get_context_from_index(self, index: int) -> str:
        """
        Get the context from the index.

        Args:
            index (`int`):
                The index of the context.

        Returns:
            `str`: The context.
        """
        if self._context_embeddings is None:
            raise ValueError(
                "The contexts must be indexed before they can be retrieved."
            )
        return self._context_index.get_label_from_index(index)

    def get_vector_from_index(self, index: int) -> torch.Tensor:
        """
        Get the context vector from the index.

        Args:
            index (`int`):
                The index of the context.

        Returns:
            `torch.Tensor`: The context vector.
        """
        if self._context_embeddings is None:
            raise ValueError(
                "The contexts must be indexed before they can be retrieved."
            )
        return self._context_embeddings[index]

    def get_vector_from_context(self, context: str) -> torch.Tensor:
        """
        Get the context vector from the context.

        Args:
            context (`str`):
                The context.

        Returns:
            `torch.Tensor`: The context vector.
        """
        if self._context_embeddings is None:
            raise ValueError(
                "The contexts must be indexed before they can be retrieved."
            )
        return self.get_vector_from_index(self.get_index_from_context(context))

    @property
    def question_tokenizer(self) -> tr.PreTrainedTokenizer:
        """
        The question tokenizer.
        """
        if self._question_tokenizer:
            return self._question_tokenizer

        if (
            self.question_encoder.language_model_name
            == self.question_encoder.language_model_name
        ):
            if not self._question_tokenizer:
                self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                    self.question_encoder.language_model_name
                )
            self._context_tokenizer = self._question_tokenizer
            return self._question_tokenizer

        if not self._question_tokenizer:
            self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                self.question_encoder.language_model_name
            )
        return self._question_tokenizer

    @property
    def context_tokenizer(self) -> tr.PreTrainedTokenizer:
        """
        The context tokenizer.
        """
        if self._context_tokenizer:
            return self._context_tokenizer

        if (
            self.question_encoder.language_model_name
            == self.context_encoder.language_model_name
        ):
            if not self._question_tokenizer:
                self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                    self.question_encoder.language_model_name
                )
            self._context_tokenizer = self._question_tokenizer
            return self._context_tokenizer

        if not self._context_tokenizer:
            self._context_tokenizer = tr.AutoTokenizer.from_pretrained(
                self.context_encoder.language_model_name
            )
        return self._context_tokenizer

    @property
    def context_embeddings(self) -> torch.Tensor:
        """
        The context embeddings.
        """
        return self._context_embeddings

    @property
    def context_index(self) -> Labels:
        """
        The context index.
        """
        return self._context_index

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
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "question_encoder": self.question_encoder.config,
            "context_encoder": self.context_encoder.config
            if not self.context_encoder_is_question_encoder
            else None,
            "loss_type": {
                "_target_": f"{self.loss_type.__class__.__module__}.{self.loss_type.__class__.__name__}"
            },
            # context_index is not saved because it might be too large
            "context_index": None,
        }

    def save_pretrained(
        self,
        output_dir: Union[str, os.PathLike],
        config: Optional[Dict[str, Any]] = None,
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

        logger.log(f"Saving retriever to {output_dir}")
        logger.log(f"Saving config to {output_dir / CONFIG_NAME}")
        # pretty print the config
        pprint(config, console=logger, expand_all=True)

        # override the from_pretrained parameter of the encoders
        # we don't want to load the pretrained weights from HF Hub when loading the retriever
        config["question_encoder"]["from_pretrained"] = False
        if config["context_encoder"] is not None:
            config["context_encoder"]["from_pretrained"] = False
        # save the config using OmegaConf
        OmegaConf.save(config, output_dir / CONFIG_NAME)

        if self._context_embeddings is None or self._context_index is None:
            raise ValueError("The contexts must be indexed before they can be saved.")

        # save the current state of the retriever
        logger.log(f"Saving retriever state to {output_dir / WEIGHTS_NAME}")
        torch.save(self.state_dict(), output_dir / WEIGHTS_NAME)
        # save the context embeddings
        logger.log(f"Saving context embeddings to {output_dir / INDEX_VECTOR_NAME}")
        torch.save(self._context_embeddings, output_dir / INDEX_VECTOR_NAME)
        # save the context index
        logger.log(f"Saving context index to {output_dir / INDEX_NAME}")
        self._context_index.save(output_dir / INDEX_NAME)
        # save the faiss indexer
        # if self._faiss_indexer is not None:
        #     logger.log(f"Saving faiss index to {output_dir / FAISS_INDEX_NAME}")
        #     self._faiss_indexer.save(output_dir)

        logger.log("Saving retriever to disk done.")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_dir: Union[str, Path],
        strict: bool = True,
        device: str = "cpu",
        index_device: str = "cpu",
        index_precision: str = "fp32",
        load_faiss_index: bool = False,
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
            load_faiss_index (`bool`, optional):
                Whether to load the faiss index. Defaults to `False`.
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
            raise ValueError(f"Providing a model id is not supported yet.")
            # model_archive = sapienzanlp_model_urls(model_name_or_dir_)

        model_dir = from_cache(
            model_archive,
            cache_dir=cache_dir,
            force_download=force_download,
        )

        logger.log(f"Loading retriever from {model_dir}")
        # get model stuff
        if device == "cpu":
            num_threads = os.getenv(
                "TORCH_NUM_THREADS", psutil.cpu_count(logical=False)
            )
            torch.set_num_threads(num_threads)
            logger.log(f"Model is running on {num_threads} threads")

        # parse config file
        config_path = model_dir / CONFIG_NAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found at {config_path}."
            )
        config = OmegaConf.load(config_path)
        pprint(OmegaConf.to_container(config), console=logger, expand_all=True)

        # load the index vocabulary
        context_index = Labels.from_file(model_dir / INDEX_NAME)

        weights_path = model_dir / WEIGHTS_NAME

        # load model from config
        logger.log("Loading model")
        model = hydra.utils.instantiate(
            config, context_index=context_index, *args, **kwargs
        )
        # load model weights
        model_state = torch.load(weights_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(
            model_state, strict=strict
        )
        if unexpected_keys or missing_keys:
            logger.log(
                f"Error loading state dict for {model.__class__.__name__}\n\t"
                f"Missing keys: {missing_keys}\n\t"
                f"Unexpected keys: {unexpected_keys}"
            )

        # run some checks
        index_vectors = model_dir / INDEX_VECTOR_NAME
        faiss_index_vectors = model_dir / FAISS_INDEX_NAME
        if load_faiss_index and not faiss_index_vectors.exists():
            logger.log(
                f"{FAISS_INDEX_NAME} does not exist. Trying to convert from dense index."
            )
        if not index_vectors.exists():
            raise ValueError(f"Index vectors `{index_vectors}` does not exist.")
        if not (model_dir / INDEX_NAME).exists():
            raise ValueError(f"Index `{model_dir / INDEX_NAME}` does not exist.")

        # select between faiss and dense index
        if load_faiss_index:
            faiss_kwargs = {
                "use_gpu": bool(index_device == "cuda"),
            }
            if not faiss_index_vectors.exists():
                # try to load the faiss index from the torch index
                embeddings = torch.load(index_vectors, map_location="cpu")
                faiss_kwargs.update({"embeddings": embeddings})
                model._faiss_indexer = FaissIndexer(**faiss_kwargs)
                del embeddings
            else:
                faiss_kwargs.update({"loading_dir": model_dir})
                model._faiss_indexer = FaissIndexer.load(**faiss_kwargs)
        else:
            logger.log("Loading index vectors")
            model._context_embeddings = torch.load(index_vectors, map_location="cpu")
            if device == "cuda":
                model._context_embeddings = model._context_embeddings.to(
                    PRECISION_MAP[index_precision]
                )
                model._context_embeddings = model._context_embeddings.to(device)

        # move model to device
        model.to(device)
        return model
