import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from utils.model_inputs import ModelInputs

from utils.logging import get_console_logger

# from faiss.indexer import FaissIndexer

logger = get_console_logger()


class PredictionCallback(pl.Callback):
    def __init__(
        self,
        batch_size: int = 32,
        output_dir: Optional[Path] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.output_dir = output_dir

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
        *args,
        **kwargs,
    ) -> Any:
        # it should return the predictions
        raise NotImplementedError

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        self(trainer, pl_module, "validation")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self(trainer, pl_module, "test")

class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: int = 100,
        report_intervals: Optional[int] = None,
        batch_size: int = 32,
        output_dir: Optional[Path] = None,
        *args,
        **kwargs,
    ):
        super().__init__(batch_size, output_dir, *args, **kwargs)
        self.k = k
        self.report_intervals = report_intervals

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing predictions for stage {stage}")

        if stage not in ["validation", "test"]:
            raise ValueError(
                f"Stage {stage} not supported, only `validation` and `test` are supported."
            )

        dataloaders = (
            trainer.val_dataloaders
            if stage == "validation"
            else trainer.test_dataloaders
        )
        datasets = (
            trainer.datamodule.val_datasets
            if stage == "validation"
            else trainer.datamodule.test_datasets
        )

        tokenizer = trainer.datamodule.tokenizer

        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            logger.log(f"Computing context embeddings for dataloader {dataloader_idx}")
            if datasets[dataloader_idx].contexts is not None:
                context_dataloader = DataLoader(
                    datasets[dataloader_idx].contexts,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=lambda x: ModelInputs(
                        {
                            # this is a hack to normalize the batch structure
                            "contexts": tokenizer(x, padding=True, return_tensors="pt")
                        }
                    ),
                )
            else:
                context_dataloader = dataloader

            context_embeddings, context_index = self.compute_context_embeddings(
                pl_module.model.context_encoder, context_dataloader, pl_module.device
            )

            # now compute the question embeddings and compute the top-k accuracy
            logger.log(f"Computing predictions for dataloader {dataloader_idx}")
            predictions = []
            for batch in dataloader:
                batch = batch.to(pl_module.device)
                model_outputs = pl_module.model(
                    batch.questions, contexts_encodings=context_embeddings
                )
                similarity = model_outputs["logits"]
                # get the top-k indices
                top_ks = torch.topk(
                    similarity, k=min(self.k, similarity.shape[-1]), dim=1
                ).indices
                # compute recall at k
                for sample_idx, top_k in enumerate(top_ks):
                    labels = batch["labels_for_metrics"][sample_idx]
                    # get the positive context ids
                    positive_context_ids = [
                        " ".join(map(str, [c for c in context_ids.tolist() if c != 0]))
                        for context_ids, label in zip(
                            batch["contexts"]["input_ids"], labels
                        )
                        if label == 1
                    ]
                    # get the top_k context ids
                    top_k_context_ids = [
                        context_index[context_idx] for context_idx in top_k.tolist()
                    ]
                    # compute the recall at k
                    positives = set(top_k_context_ids) & set(positive_context_ids)
                    negatives = set(top_k_context_ids) - set(positive_context_ids)
                    # convert the context ids to text
                    positives = [
                        tokenizer.decode(
                            list(map(int, context_id.split(" "))),
                            skip_special_tokens=True,
                        )
                        for context_id in positives
                    ]
                    negatives = [
                        tokenizer.decode(
                            list(map(int, context_id.split(" "))),
                            skip_special_tokens=True,
                        )
                        for context_id in negatives
                    ]
                    # convert top-k to text too
                    top_k_contexts = [
                        tokenizer.decode(
                            list(map(int, context_id.split(" "))),
                            skip_special_tokens=True,
                        )
                        for context_id in top_k_context_ids
                    ]
                    predictions.append(
                        {
                            "question": tokenizer.decode(
                                batch["questions"]["input_ids"][sample_idx],
                                skip_special_tokens=True,
                            ),
                            "top_k_contexts": top_k_contexts,
                            "correct_contexts": positives,
                            "wrong_contexts": negatives,
                        }
                    )
        # write the predictions to a file inside the experiment folder
        if self.output_dir is None and trainer.logger is None:
            raise ValueError(
                "You need to specify an output directory or a logger to save the predictions."
            )

        if self.output_dir is not None:
            prediction_folder = self.output_dir
        else:
            prediction_folder = Path(trainer.logger.experiment.dir) / "predictions"
            prediction_folder.mkdir(exist_ok=True)
        prediction_file = (
            prediction_folder / f"{stage}_dataloader_{dataloader_idx}.json"
        )
        logger.log(f"Writing predictions to {prediction_file}")
        with open(prediction_file, "w") as f:
            json.dump(predictions, f, indent=2)

    @staticmethod
    @torch.no_grad()
    def compute_context_embeddings(
        context_encoder: torch.nn.Module,
        dataloader: DataLoader,
        device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, Dict[int, str]]:
        # Create empty lists to store the context embeddings and context index
        context_embeddings: List[torch.Tensor] = []
        context_index: Dict[int, str] = {}
        # Create an empty set to keep track of the contexts that have already been seen
        already_seen: Set[str] = set()
        # index to keep track of the contexts
        i: int = 0
        # Iterate through each batch in the dataloader
        for batch in dataloader:
            # Move the batch to the device
            batch = batch.to(device)
            # Compute the context embeddings
            context_outs = context_encoder(**batch.contexts)
            # Loop through each context in the batch
            for context_ids, e in zip(batch.contexts.input_ids, context_outs):
                # Clean up the context by removing any 0s
                cleaned_context = " ".join(
                    map(str, [c for c in context_ids.tolist() if c != 0])
                )
                # If the cleaned context has not been seen, add it to the empty lists and set
                if cleaned_context not in already_seen:
                    already_seen.add(cleaned_context)
                    context_embeddings.append(e)
                    context_index[i] = cleaned_context
                    i += 1
        # Stack the context embeddings into a tensor and return it along with the context index
        context_embeddings = torch.stack(context_embeddings, dim=0)
        return context_embeddings, context_index
