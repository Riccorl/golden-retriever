import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import hydra
import psutil
import torch
import torch.nn.functional as F
import transformers as tr
from omegaconf import OmegaConf
from rich.pretty import pprint

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
from goldenretriever.data.labels import Labels
from goldenretriever.retriever import RetrievedSample
from goldenretriever.retriever.indexers.base import BaseDocumentIndex
from goldenretriever.retriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.retriever.modules.encoder.hf import (
    GoldenRetrieverModel,
)
from goldenretriever.retriever.modules.encoder.torch import SentenceEncoder

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


class GoldenRetrieverHF(torch.nn.Module):
    def __init__(
        self,
        question_encoder: Union[str, tr.PreTrainedModel],
        loss_type: Optional[torch.nn.Module] = None,
        passage_encoder: Optional[Union[str, tr.PreTrainedModel]] = None,
        document_index: Optional[Union[str, BaseDocumentIndex]] = None,
        question_tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        passage_tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.passage_encoder_is_question_encoder = False
        # question encoder model
        if isinstance(question_encoder, str):
            question_encoder = GoldenRetrieverModel.from_pretrained(question_encoder)
        self.question_encoder = question_encoder
        if passage_encoder is None:
            # if no passage encoder is provided,
            # share the weights of the question encoder
            passage_encoder = question_encoder
            # keep track of the fact that the passage encoder is the same as the question encoder
            self.passage_encoder_is_question_encoder = True
        if isinstance(passage_encoder, str):
            passage_encoder = GoldenRetrieverModel.from_pretrained(passage_encoder)
        # passage encoder model
        self.passage_encoder = passage_encoder

        # loss function
        self.loss_type = loss_type

        # indexer stuff
        if isinstance(document_index, str):
            document_index = BaseDocumentIndex.from_pretrained(document_index)
        self.document_index = document_index

        # lazy load the tokenizer for inference
        self._question_tokenizer = question_tokenizer
        self._passage_tokenizer = passage_tokenizer

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
            question_encodings = self.question_encoder(**questions)
        if passages_encodings is None:
            passages_encodings = self.passage_encoder(**passages)

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
        # passages: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        force_reindex: bool = False,
        use_ort: bool = False,
        compute_on_cpu: bool = False,
        precision: Optional[Union[str, int]] = None,
    ):
        """
        Index the passages for later retrieval.

        Args:
            batch_size (`int`):
                The batch size to use for the indexing.
            num_workers (`int`):
                The number of workers to use for the indexing.
            max_length (`Optional[int]`):
                The maximum length of the passages.
            collate_fn (`Callable`):
                The collate function to use for the indexing.
            force_reindex (`bool`):
                Whether to force reindexing even if the passages are already indexed.
            use_ort (`bool`):
                Whether to use onnxruntime for the indexing.
            move_index_to_cpu (`bool`):
                Whether to move the index to the CPU after the indexing.
            precision (`Optional[Union[str, int]]`):
                The precision to use for the model.
            index_precision (`Optional[Union[str, int]]`):
                The precision to use for the index.
        """
        if self.document_index is None:
            raise ValueError(
                "The retriever must be initialized with an indexer to index the passages within the retriever."
            )
        return self.document_index.index(
            retriever=self,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            collate_fn=collate_fn,
            encoder_precision=precision,
            compute_on_cpu=compute_on_cpu,
            force_reindex=force_reindex,
        )

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
        if self.document_index is None:
            raise ValueError(
                "The indexer must be indexed before it can be used within the retriever."
            )
        if text is None and input_ids is None:
            raise ValueError(
                "Either `text` or `input_ids` must be provided to retrieve the passages."
            )

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

        # fucking autocast only wants pure strings like 'cpu' or 'cuda'
        # we need to convert the model device to that
        device_type_for_autocast = str(self.device).split(":")[0]
        # autocast doesn't work with CPU and stuff different from bfloat16
        autocast_pssg_mngr = (
            contextlib.nullcontext()
            if device_type_for_autocast == "cpu"
            else (
                torch.autocast(
                    device_type=device_type_for_autocast,
                    dtype=PRECISION_MAP[precision],
                )
            )
        )
        with autocast_pssg_mngr:
            question_encodings = self.question_encoder(**model_inputs)

        return self.document_index.search(question_encodings, k)

    def get_index_from_passage(self, passage: str) -> int:
        """
        Get the index of the passage.

        Args:
            passage (`str`):
                The passage to get the index for.

        Returns:
            `int`: The index of the passage.
        """
        if self.document_index is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.document_index.get_index_from_passage(passage)

    def get_passage_from_index(self, index: int) -> str:
        """
        Get the passage from the index.

        Args:
            index (`int`):
                The index of the passage.

        Returns:
            `str`: The passage.
        """
        if self.document_index is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.document_index.get_passage_from_index(index)

    def get_vector_from_index(self, index: int) -> torch.Tensor:
        """
        Get the passage vector from the index.

        Args:
            index (`int`):
                The index of the passage.

        Returns:
            `torch.Tensor`: The passage vector.
        """
        if self.document_index is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.document_index.get_embeddings_from_index(index)

    def get_vector_from_passage(self, passage: str) -> torch.Tensor:
        """
        Get the passage vector from the passage.

        Args:
            passage (`str`):
                The passage.

        Returns:
            `torch.Tensor`: The passage vector.
        """
        if self.document_index is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.document_index.get_embeddings_from_passage(passage)

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

    @property
    def question_tokenizer(self) -> tr.PreTrainedTokenizer:
        """
        The question tokenizer.
        """
        if self._question_tokenizer:
            return self._question_tokenizer

        if (
            self.question_encoder.config.name_or_path
            == self.question_encoder.config.name_or_path
        ):
            if not self._question_tokenizer:
                self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                    self.question_encoder.config.name_or_path
                )
            self._passage_tokenizer = self._question_tokenizer
            return self._question_tokenizer

        if not self._question_tokenizer:
            self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                self.question_encoder.config.name_or_path
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
            self.question_encoder.config.name_or_path
            == self.passage_encoder.config.name_or_path
        ):
            if not self._question_tokenizer:
                self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                    self.question_encoder.config.name_or_path
                )
            self._passage_tokenizer = self._question_tokenizer
            return self._passage_tokenizer

        if not self._passage_tokenizer:
            self._passage_tokenizer = tr.AutoTokenizer.from_pretrained(
                self.passage_encoder.config.name_or_path
            )
        return self._passage_tokenizer

    def save_pretrained(
        self,
        output_dir: Union[str, os.PathLike],
        question_encoder_name: Optional[str] = None,
        passage_encoder_name: Optional[str] = None,
        document_index_name: Optional[str] = None,
        push_to_hub: bool = False,
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

        # create the output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving retriever to {output_dir}")

        question_encoder_name = question_encoder_name or "question_encoder"
        passage_encoder_name = passage_encoder_name or "passage_encoder"
        document_index_name = document_index_name or "document_index"

        logger.info(
            f"Saving question encoder state to {output_dir / question_encoder_name}"
        )
        self.question_encoder.save_pretrained(output_dir / question_encoder_name)
        self.question_tokenizer.save_pretrained(output_dir / question_encoder_name)
        if self.passage_encoder is not None:
            logger.info(
                f"Saving passage encoder state to {output_dir / passage_encoder_name}"
            )
            self.passage_encoder.save_pretrained(output_dir / passage_encoder_name)
            self.passage_tokenizer.save_pretrained(output_dir / passage_encoder_name)

        if self.document_index is not None:
            # save the indexer
            self.document_index.save_pretrained(output_dir / document_index_name)

        logger.info("Saving retriever to disk done.")


class GoldenRetriever(torch.nn.Module):
    def __init__(
        self,
        question_encoder: SentenceEncoder,
        loss_type: Optional[torch.nn.Module] = None,
        passage_encoder: Optional[SentenceEncoder] = None,
        document_index: Optional[BaseDocumentIndex] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.passage_encoder_is_question_encoder = False
        # question encoder model
        self.question_encoder = question_encoder
        if passage_encoder is None:
            # if no passage encoder is provided,
            # share the weights of the question encoder
            passage_encoder = question_encoder
            # keep track of the fact that the passage encoder is the same as the question encoder
            self.passage_encoder_is_question_encoder = True
        # passage encoder model
        self.passage_encoder = passage_encoder

        # loss function
        self.loss_type = loss_type

        # indexer stuff
        self.document_index = document_index

        # lazy load the tokenizer for inference
        self._question_tokenizer: Optional[tr.PreTrainedTokenizer] = None
        self._passage_tokenizer: Optional[tr.PreTrainedTokenizer] = None

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
            question_encodings = self.question_encoder(**questions)
        if passages_encodings is None:
            passages_encodings = self.passage_encoder(**passages)

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
        # passages: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        force_reindex: bool = False,
        use_ort: bool = False,
        compute_on_cpu: bool = False,
        precision: Optional[Union[str, int]] = None,
    ):
        """
        Index the passages for later retrieval.

        Args:
            batch_size (`int`):
                The batch size to use for the indexing.
            num_workers (`int`):
                The number of workers to use for the indexing.
            max_length (`Optional[int]`):
                The maximum length of the passages.
            collate_fn (`Callable`):
                The collate function to use for the indexing.
            force_reindex (`bool`):
                Whether to force reindexing even if the passages are already indexed.
            use_ort (`bool`):
                Whether to use onnxruntime for the indexing.
            move_index_to_cpu (`bool`):
                Whether to move the index to the CPU after the indexing.
            precision (`Optional[Union[str, int]]`):
                The precision to use for the model.
            index_precision (`Optional[Union[str, int]]`):
                The precision to use for the index.
        """
        if self.document_index is None:
            raise ValueError(
                "The retriever must be initialized with an indexer to index the passages within the retriever."
            )
        return self.document_index.index(
            retriever=self,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            collate_fn=collate_fn,
            encoder_precision=precision,
            compute_on_cpu=compute_on_cpu,
            force_reindex=force_reindex,
        )

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
        if self.document_index is None:
            raise ValueError(
                "The indexer must be indexed before it can be used within the retriever."
            )
        if text is None and input_ids is None:
            raise ValueError(
                "Either `text` or `input_ids` must be provided to retrieve the passages."
            )

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

        # fucking autocast only wants pure strings like 'cpu' or 'cuda'
        # we need to convert the model device to that
        device_type_for_autocast = str(self.device).split(":")[0]
        # autocast doesn't work with CPU and stuff different from bfloat16
        autocast_pssg_mngr = (
            contextlib.nullcontext()
            if device_type_for_autocast == "cpu"
            else (
                torch.autocast(
                    device_type=device_type_for_autocast,
                    dtype=PRECISION_MAP[precision],
                )
            )
        )
        with autocast_pssg_mngr:
            question_encodings = self.question_encoder(**model_inputs)

        return self.document_index.search(question_encodings, k)

    def get_index_from_passage(self, passage: str) -> int:
        """
        Get the index of the passage.

        Args:
            passage (`str`):
                The passage to get the index for.

        Returns:
            `int`: The index of the passage.
        """
        if self.document_index is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.document_index.get_index_from_passage(passage)

    def get_passage_from_index(self, index: int) -> str:
        """
        Get the passage from the index.

        Args:
            index (`int`):
                The index of the passage.

        Returns:
            `str`: The passage.
        """
        if self.document_index is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.document_index.get_passage_from_index(index)

    def get_vector_from_index(self, index: int) -> torch.Tensor:
        """
        Get the passage vector from the index.

        Args:
            index (`int`):
                The index of the passage.

        Returns:
            `torch.Tensor`: The passage vector.
        """
        if self.document_index is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.document_index.get_embeddings_from_index(index)

    def get_vector_from_passage(self, passage: str) -> torch.Tensor:
        """
        Get the passage vector from the passage.

        Args:
            passage (`str`):
                The passage.

        Returns:
            `torch.Tensor`: The passage vector.
        """
        if self.document_index is None:
            raise ValueError(
                "The passages must be indexed before they can be retrieved."
            )
        return self.document_index.get_embeddings_from_passage(passage)

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
            self._passage_tokenizer = self._question_tokenizer
            return self._question_tokenizer

        if not self._question_tokenizer:
            self._question_tokenizer = tr.AutoTokenizer.from_pretrained(
                self.question_encoder.language_model_name
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
            question_encoder=self.question_encoder.config,
            passage_encoder=self.passage_encoder.config
            if not self.passage_encoder_is_question_encoder
            else None,
            loss_type=dict(
                _target_=f"{self.loss_type.__class__.__module__}.{self.loss_type.__class__.__name__}"
            ),
        )

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

        logger.info(f"Saving retriever to {output_dir}")
        logger.info(f"Saving config to {output_dir / CONFIG_NAME}")
        # pretty print the config
        pprint(config, console=console_logger, expand_all=True)

        # override the from_pretrained parameter of the encoders
        # we don't want to load the pretrained weights from HF Hub when loading the retriever
        config["question_encoder"]["from_pretrained"] = False
        if config["passage_encoder"] is not None:
            config["passage_encoder"]["from_pretrained"] = False
        # save the config using OmegaConf
        OmegaConf.save(config, output_dir / CONFIG_NAME)

        # save the current state of the retriever
        logger.info(f"Saving retriever state to {output_dir / WEIGHTS_NAME}")
        torch.save(self.state_dict(), output_dir / WEIGHTS_NAME)
        if self.document_index is not None:
            # save the indexer
            self.document_index.save(output_dir)

        logger.info("Saving retriever to disk done.")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_dir: Union[str, Path],
        strict: bool = True,
        device: str = "cpu",
        index_device: Optional[str] = None,
        index_precision: Optional[str] = None,
        load_index_vector: bool = True,
        index_type: BaseDocumentIndex = InMemoryDocumentIndex,
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

        weights_path = model_dir / WEIGHTS_NAME

        # load model from config
        logger.info("Loading model")
        model = hydra.utils.instantiate(config, *args, **kwargs)
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
            index_device = index_device or device
            model.document_index = index_type.load(
                model_dir, device=index_device, precision=index_precision
            )

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
