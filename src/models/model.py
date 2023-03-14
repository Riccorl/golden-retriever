import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers as tr
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer
from optimum.onnxruntime.configuration import (
    AutoOptimizationConfig,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import BaseDataset
from data.labels import Labels
from models.faiss_indexer import FaissIndexer
from utils.logging import get_console_logger
from utils.model_inputs import ModelInputs

logger = get_console_logger()


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        language_model: Union[
            str, tr.PreTrainedModel, ORTModelForFeatureExtraction
        ] = "sentence-transformers/all-MiniLM-6-v2",
        pooling_strategy: str = "mean",
        load_ort_model: bool = False,
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
            # normalize
            normalized = F.normalize(mean_pooling, p=2, dim=1)
            return normalized
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
        labels: Optional[Labels] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if labels is not None:
            self.labels = labels

        # question encoder model
        self.question_encoder = question_encoder
        if not context_encoder:
            # if no context encoder is provided,
            # share the weights of the question encoder
            context_encoder = question_encoder
        # context encoder model
        self.context_encoder = context_encoder
        # loss function
        self.loss_type = loss_type

        # indexer stuff, lazy loaded
        self._context_embeddings: Optional[torch.Tensor] = None
        self._context_index: Optional[Dict[int, str]] = None
        self._reverse_context_index: Optional[Dict[str, int]] = None
        self.faiss_indexer: Optional[FaissIndexer] = None

    def forward(
        self,
        questions: Optional[Dict[str, torch.Tensor]] = None,
        contexts: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        question_encodings: Optional[torch.Tensor] = None,
        contexts_encodings: Optional[torch.Tensor] = None,
        contexs_per_question: Optional[List[int]] = None,
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

        if contexs_per_question is not None:
            # multiply each question encoding with a contexs_per_question encodings
            concatenated_contexts = torch.stack(
                torch.split(contexts_encodings, contexs_per_question)
            ).transpose(1, 2)
            logits = torch.bmm(
                question_encodings.unsqueeze(1), concatenated_contexts
            # ).view(contexts_encodings.shape[0], -1)
            ).squeeze(1)
        else:
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

    def index(
        self,
        contexts: List[str],
        batch_size: int = 32,
        num_workers: int = 8,
        collate_fn: Optional[Callable] = None,
        force_reindex: bool = False,
        use_faiss: bool = False,
        use_gpu: bool = False,
        use_ort: bool = False,
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
            collate_fn (`Callable`):
                The collate function to use for the indexing.
            force_reindex (`bool`):
                Whether to force reindexing even if the contexts are already indexed.
            use_faiss (`bool`):
                Whether to use faiss for the indexing.
            use_gpu (`bool`):
                Whether to use the GPU for the indexing.
            use_ort (`bool`):
                Whether to use onnxruntime for the indexing.
        """
        if self._context_embeddings is not None and not force_reindex:
            return

        if collate_fn is None:
            tokenizer = tr.AutoTokenizer.from_pretrained(
                self.context_encoder.language_model.pretrained_model_name_or_path
            )
            collate_fn = lambda x: ModelInputs(
                {
                    tokenizer(
                        x,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.model.max_length,
                    )
                }
            )
        dataloader = DataLoader(
            BaseDataset(name="context", data=contexts),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        # we can use the onnx runtime optimized encoder for the indexing
        if not use_ort:
            context_encoder = self.context_encoder
        else:
            context_encoder = self._load_ort_optimized_encoder(self.context_encoder)
        # Create empty lists to store the context embeddings and context index
        context_embeddings: List[torch.Tensor] = []
        # Iterate through each batch in the dataloader
        for batch in tqdm(dataloader, desc="Indexing"):
            # Move the batch to the device
            batch: ModelInputs = batch.to(next(self.parameters()).device)
            # Compute the context embeddings
            context_outs = context_encoder(**batch)
            # Append the context embeddings to the list
            context_embeddings.extend([c if use_gpu else c.cpu() for c in context_outs])

        # Stack the context embeddings into a tensor and return it along with the context index
        self._context_embeddings = None
        self._context_embeddings = torch.stack(context_embeddings, dim=0)
        # Create a dictionary mapping the context index to the context
        self._context_index = {i: context for i, context in enumerate(contexts)}
        # reverse the context index
        self._reverse_context_index = {v: k for k, v in self._context_index.items()}
        if use_faiss and self.faiss_indexer is None:
            self.faiss_indexer = FaissIndexer(self._context_embeddings, use_gpu=use_gpu)

    def retrieve(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        k: int = 5,
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """
        Retrieve the contexts for the questions.

        Args:
            input_ids (`torch.Tensor`):
                The input ids of the questions.
            attention_mask (`torch.Tensor`):
                The attention mask of the questions.
            token_type_ids (`torch.Tensor`):
                The token type ids of the questions.
            k (`int`):
                The number of top contexts to retrieve.

        Returns:
            `Tuple[List[List[str]], List[List[int]]]`: The retrieved contexts and their indices.
        """
        if self._context_embeddings is None:
            raise ValueError(
                "The contexts must be indexed before they can be retrieved."
            )
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        if self.faiss_indexer is not None:
            top_k: torch.Tensor = self.faiss_indexer.search(
                self.question_encoder(**model_inputs), k=k
            )
        else:
            model_inputs = {
                "questions": model_inputs,
                "contexts_encodings": self._context_embeddings,
            }
            similarity = self(**model_inputs)["logits"]
            # Retrieve the indices of the top k context embeddings
            top_k: torch.Tensor = torch.topk(
                similarity, k=min(k, similarity.shape[-1]), dim=1
            ).indices
        # get int values
        top_k: List[List[int]] = top_k.cpu().tolist()
        # Retrieve the contexts corresponding to the indices
        contexts = [[self._context_index[i] for i in indices] for indices in top_k]
        return contexts, top_k

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
        if context not in self._reverse_context_index:
            raise ValueError(
                f"The context '{context}' is not in the index. Please index the context before retrieving it."
            )
        return self._reverse_context_index[context]

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
        if index not in self._context_index:
            raise ValueError(
                f"The index '{index}' is not in the index. Please index the context before retrieving it."
            )
        return self._context_index[index]

    @property
    def context_embeddings(self) -> torch.Tensor:
        """
        The context embeddings.
        """
        return self._context_embeddings

    @property
    def context_index(self) -> Dict[int, str]:
        """
        The context index.
        """
        return self._context_index

    @staticmethod
    def _load_ort_optimized_encoder(
        encoder: SentenceEncoder, provider: str = "CUDAExecutionProvider"
    ) -> SentenceEncoder:
        temp_dir = tempfile.mkdtemp()
        encoder.language_model.save_pretrained(temp_dir)
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            temp_dir, export=True, provider=provider, use_io_binding=True
        )
        optimizer = ORTOptimizer.from_pretrained(ort_model)
        optimization_config = AutoOptimizationConfig.O4(optimization_level=99)
        optimizer.optimize(save_dir=temp_dir, optimization_config=optimization_config)
        # quantizer = ORTQuantizer.from_pretrained(temp_dir)
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            temp_dir, export=True, provider=provider, use_io_binding=True
        )
        return SentenceEncoder(
            language_model=ort_model,
            pooling_strategy=encoder.pooling_strategy,
        )
