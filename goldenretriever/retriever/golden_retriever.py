import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import transformers as tr

from goldenretriever.common.log import get_console_logger, get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.common.utils import is_package_available
from goldenretriever.data.labels import Labels
from goldenretriever.retriever import RetrievedSample
from goldenretriever.retriever.indexers.base import BaseDocumentIndex
from goldenretriever.retriever.modules.hf import GoldenRetrieverModel


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


class GoldenRetriever(torch.nn.Module):
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
            question_encodings = self.question_encoder(**questions).pooler_output
        if passages_encodings is None:
            passages_encodings = self.passage_encoder(**passages).pooler_output

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
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        force_reindex: bool = False,
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
            question_encodings = self.question_encoder(**model_inputs).pooler_output

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
        **kwargs,
    ):
        """
        Save the retriever to a directory.

        Args:
            output_dir (`str`):
                The directory to save the retriever to.
            question_encoder_name (`Optional[str]`):
                The name of the question encoder.
            passage_encoder_name (`Optional[str]`):
                The name of the passage encoder.
            document_index_name (`Optional[str]`):
                The name of the document index.
            push_to_hub (`bool`):
                Whether to push the model to the hub.
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
        # self.question_encoder.config._name_or_path = question_encoder_name
        self.question_encoder.register_for_auto_class()
        self.question_encoder.save_pretrained(
            output_dir / question_encoder_name, push_to_hub=push_to_hub, **kwargs
        )
        self.question_tokenizer.save_pretrained(
            output_dir / question_encoder_name, push_to_hub=push_to_hub, **kwargs
        )
        if not self.passage_encoder_is_question_encoder:
            logger.info(
                f"Saving passage encoder state to {output_dir / passage_encoder_name}"
            )
            # self.passage_encoder.config._name_or_path = passage_encoder_name
            self.passage_encoder.register_for_auto_class()
            self.passage_encoder.save_pretrained(
                output_dir / passage_encoder_name, push_to_hub=push_to_hub, **kwargs
            )
            self.passage_tokenizer.save_pretrained(
                output_dir / passage_encoder_name, push_to_hub=push_to_hub, **kwargs
            )

        if self.document_index is not None:
            # save the indexer
            self.document_index.save_pretrained(output_dir / document_index_name, **kwargs)

        logger.info("Saving retriever to disk done.")
