from collections import defaultdict
import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Set, Union, Tuple

import hydra
import pytorch_lightning as pl
import torch
import transformers as tr
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from callbacks.base import PredictionCallback, Stage
from data.datasets import BaseDataset, DPRDataset
from models.model import GoldenRetriever
from utils.logging import get_console_logger
from utils.model_inputs import ModelInputs

logger = get_console_logger()


class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: int = 100,
        batch_size: int = 32,
        num_workers: int = 0,
        use_faiss: bool = False,
        move_index_to_cpu: bool = True,
        force_reindex: bool = True,
        retriever_dir: Optional[Path] = None,
        save_predictions: bool = True,
        predictions_dir: Optional[Path] = None,
        remove_columns: Optional[List[str]] = None,
        stages: Set[Union[str, Stage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        dataloader: Optional[DataLoader] = None,
        *args,
        **kwargs,
    ):
        super().__init__(stages, other_callbacks)
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_faiss = use_faiss
        self.move_index_to_cpu = move_index_to_cpu
        self.force_reindex = force_reindex
        self.retriever_dir = retriever_dir
        self.save_predictions = save_predictions
        self.predictions_dir = predictions_dir
        self.dataset = dataset
        self.dataloader = dataloader

        if remove_columns is None:
            remove_columns = [
                "context",
                "positives",
                "negatives",
                "wrong",
                "positive_ctxs",
                "negative_ctxs",
                "hard_negative_ctxs",
                "positive_index_end",
            ]
        self.remove_columns = remove_columns

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        *args,
        **kwargs,
    ) -> dict:
        logger.log(f"Computing predictions for stage {stage.value}")

        if stage not in self.stages:
            raise ValueError(
                f"Stage {stage} not supported, only {self.stages} are supported"
            )

        # get the tokenizer
        tokenizer = trainer.datamodule.tokenizer

        datasets, dataloaders = self._get_datasets_and_dataloaders(
            self.dataset,
            self.dataloader,
            self.batch_size,
            self.num_workers,
            stage,
            trainer,
            tokenizer,
        )

        # set the model to eval mode
        pl_module.eval()
        # get the retriever
        retriever: GoldenRetriever = pl_module.model

        # here we will store the samples with predictions for each dataloader
        dataloader_predictions = {}
        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            logger.log(
                f"Computing context embeddings for dataset {datasets[dataloader_idx].name}"
            )
            contexts = self._get_contexts_dataloader(datasets[dataloader_idx], trainer)

            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    truncation=True,
                    padding=True,
                    max_length=datasets[dataloader_idx].max_context_length,
                    return_tensors="pt",
                )
            )
            use_gpu = (pl_module.device.type == "cuda") if not self.use_faiss else False
            if self.retriever_dir is not None and trainer.current_epoch == 0:
                force_reindex = False
            else:
                force_reindex = self.force_reindex
            retriever.index(
                contexts,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                force_reindex=force_reindex,
                use_faiss=self.use_faiss,
                use_gpu=use_gpu,
                move_index_to_cpu=self.move_index_to_cpu,
            )

            pl_module_original_device = pl_module.device
            if (
                not self.use_faiss
                and self.move_index_to_cpu
                and pl_module.device.type == "cuda"
            ):
                pl_module.to("cpu")

            # now compute the question embeddings and compute the top-k accuracy
            logger.log(
                f"Computing predictions for dataset {datasets[dataloader_idx].name}"
            )
            predictions = []
            start = time.time()
            for batch in tqdm(dataloader, desc="Retrieving contexts"):
                batch = batch.to(pl_module.device)
                # get the top-k indices
                retrieved_contexts, retrieved_indices = retriever.retrieve(
                    **batch.questions, k=self.k
                )
                # compute recall at k
                for sample_idx, retrieved_index in enumerate(retrieved_indices):
                    # get the positive contexts
                    gold_contexts = batch.positives[sample_idx]
                    # get the index of the gold contexts in the retrieved contexts
                    gold_context_indices = [
                        retriever.get_index_from_context(context)
                        for context in gold_contexts
                    ]
                    # correct predictions are the contexts that are in the top-k and are gold
                    correct_indices = set(gold_context_indices) & set(retrieved_index)
                    # wrong predictions are the contexts that are in the top-k and are not gold
                    wrong_indices = set(retrieved_index) - set(gold_context_indices)
                    # add the predictions to the list
                    predictions.append(
                        {
                            "sample_idx": batch.sample_idx[sample_idx],
                            "gold": gold_contexts,
                            "predictions": retrieved_contexts[sample_idx],
                            "correct": [
                                retriever.get_context_from_index(i)
                                for i in correct_indices
                            ],
                            "wrong": [
                                retriever.get_context_from_index(i)
                                for i in wrong_indices
                            ],
                        }
                    )
            end = time.time()
            logger.log("Time to retrieve:", end - start)
            dataloader_predictions[dataloader_idx] = predictions

            if pl_module_original_device != pl_module.device:
                pl_module.to(pl_module_original_device)

            # update the dataset with the predictions
            for sample, prediction in zip(datasets[dataloader_idx], predictions):
                sample["gold"] = prediction["gold"]
                sample["predictions"] = prediction["predictions"]
                sample["correct"] = prediction["correct"]
                sample["wrong"] = prediction["wrong"]

            # write the predictions to a file inside the experiment folder
            if self.predictions_dir is None and trainer.logger is None:
                logger.log(
                    "You need to specify an output directory (`predictions_dir`) or a logger to save the predictions."
                )
            else:
                # save to file
                if self.save_predictions:
                    if self.predictions_dir is not None:
                        prediction_folder = Path(self.predictions_dir)
                    else:
                        prediction_folder = (
                            Path(trainer.logger.experiment.dir) / "predictions"
                        )
                        prediction_folder.mkdir(exist_ok=True)
                    predictions_path = (
                        prediction_folder
                        / f"{datasets[dataloader_idx].name}_{dataloader_idx}.json"
                    )
                    logger.log(f"Saving predictions to {predictions_path}")
                    datasets[dataloader_idx].save_data(
                        predictions_path, remove_columns=self.remove_columns
                    )

        # return the predictions
        return dataloader_predictions

    @staticmethod
    def _get_contexts_dataloader(dataset, trainer):
        if dataset.contexts is None:
            logger.log(
                f"Contexts not found in dataset {dataset.name}, computing them from the dataloaders"
            )
            # get the contexts from the all the dataloader context ids
            contexts = set()  # set to avoid duplicates
            for batch in trainer.train_dataloader:
                contexts.update(
                    [
                        " ".join(map(str, [c for c in context_ids.tolist() if c != 0]))
                        for context_ids in batch.contexts.input_ids
                    ]
                )
            for d in trainer.val_dataloaders:
                for batch in d:
                    contexts.update(
                        [
                            " ".join(
                                map(str, [c for c in context_ids.tolist() if c != 0])
                            )
                            for context_ids in batch.contexts.input_ids
                        ]
                    )
            for d in trainer.test_dataloaders:
                for batch in d:
                    contexts.update(
                        [
                            " ".join(
                                map(str, [c for c in context_ids.tolist() if c != 0])
                            )
                            for context_ids in batch.contexts.input_ids
                        ]
                    )
            contexts = list(contexts)
        else:
            contexts = dataset.contexts
        return contexts

    @staticmethod
    def _get_datasets_and_dataloaders(
        dataset: Optional[Union[Dataset, DictConfig]],
        dataloader: Optional[DataLoader],
        batch_size: int,
        num_workers: int,
        stage: Stage,
        trainer: pl.Trainer,
        tokenizer: tr.PreTrainedTokenizer,
    ) -> Tuple[List[DPRDataset], List[DataLoader]]:
        """
        Get the datasets and dataloaders from the datamodule or from the dataset provided.

        Args:
            dataset (`Optional[Union[Dataset, DictConfig]]`):
                The dataset to use. If `None`, the datamodule is used.
            batch_size (`int`):
                The batch size to use for the dataloaders.
            num_workers (`int`):
                The number of workers to use for the dataloaders.
            stage (`Stage`):
                The stage that indicates whether the dataloaders are for validation or testing.
            trainer (`pl.Trainer`):
                The trainer that contains the datamodule.
            tokenizer (`tr.PreTrainedTokenizer`):
                The tokenizer to use for the dataloaders.

        Returns:
            `Tuple[List[Dataset], List[DataLoader]]`: The datasets and dataloaders.
        """
        # if a dataset is provided, use it
        if dataset is not None:
            # get dataset
            if isinstance(dataset, DictConfig):
                dataset = hydra.utils.instantiate(dataset, _recursive_=False)
            datasets = [dataset]
            if dataloader is not None:
                dataloaders = [dataloader]
            else:
                dataloaders = [
                    DataLoader(
                        datasets[0],
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=partial(datasets[0].collate_fn, tokenizer=tokenizer),
                    )
                ]
        else:
            # get the dataloaders and datasets from the datamodule
            datasets = (
                trainer.datamodule.val_datasets
                if stage == Stage.VALIDATION
                else trainer.datamodule.test_datasets
            )
            dataloaders = (
                trainer.val_dataloaders
                if stage == Stage.VALIDATION
                else trainer.test_dataloaders
            )
        return datasets, dataloaders


class NegativeAugmentationCallback(GoldenRetrieverPredictionCallback):
    def __init__(
        self,
        k: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        use_faiss: bool = False,
        move_index_to_cpu: bool = False,
        force_reindex: bool = False,
        save_retriever: bool = False,
        retriever_dir: Optional[Path] = None,
        save_predictions: bool = False,
        predictions_dir: Optional[Path] = None,
        remove_columns: Optional[List[str]] = None,
        stages: Set[Union[str, Stage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        metric_to_monitor: str = "val_loss",
        threshold: float = 0.8,
        max_negatives: int = 5,
        refresh_every_n_epochs: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(
            k=k,
            batch_size=batch_size,
            num_workers=num_workers,
            use_faiss=use_faiss,
            move_index_to_cpu=move_index_to_cpu,
            force_reindex=force_reindex,
            save_retriever=save_retriever,
            retriever_dir=retriever_dir,
            predictions_dir=predictions_dir,
            save_predictions=save_predictions,
            remove_columns=remove_columns,
            stages=stages,
            other_callbacks=other_callbacks,
            dataset=dataset,
            *args,
            **kwargs,
        )
        self.metric_to_monitor = metric_to_monitor
        self.threshold = threshold
        self.max_negatives = max_negatives
        self.refresh_every_n_epochs = refresh_every_n_epochs

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Union[str, Stage],
        *args,
        **kwargs,
    ) -> dict:
        if stage not in self.stages:
            return {}

        if self.metric_to_monitor not in trainer.logged_metrics:
            raise ValueError(
                f"Metric {self.metric_to_monitor} not found in trainer.logged_metrics"
                f"Available metrics: {trainer.logged_metrics.keys()}"
            )
        if trainer.logged_metrics[self.metric_to_monitor] < self.threshold:
            return {}

        if trainer.current_epoch % self.refresh_every_n_epochs != 0:
            return {}

        logger.log(
            f"Metric {self.metric_to_monitor} is above threshold {self.threshold}. Computing hard negatives."
        )

        self.dataset = trainer.datamodule.train_dataset
        self.dataloader = trainer.datamodule.train_dataloader()

        predictions = super().__call__(trainer, pl_module, stage, *args, **kwargs)
        for dataloader_idx, prediction_samples in predictions.items():
            # store the predictions in a dictionary for faster access based on the sample index
            update_dict = defaultdict(lambda: defaultdict(list))
            for prediction in prediction_samples:
                top_k_contexts = prediction["predictions"]
                gold_contexts = prediction["gold"]
                # get the ids of the max_negatives wrong contexts with the highest similarity
                wrong_contexts = [
                    context_id
                    for context_id in top_k_contexts
                    if context_id not in gold_contexts
                ][: self.max_negatives]
                wrong_contexts_ids = trainer.datamodule.tokenizer(
                    wrong_contexts,
                    max_length=trainer.datamodule.train_dataset.max_context_length,
                    truncation=True,
                )
                retrieved_hard_negatives = []
                # for wrong_context in wrong_contexts:
                #     retrieved_hard_negatives["context"].append(wrong_context)
                for c_index in range(len(wrong_contexts)):
                    p_dict = {
                        "input_ids": wrong_contexts_ids["input_ids"][c_index],
                        "attention_mask": wrong_contexts_ids["attention_mask"][c_index],
                    }
                    if "token_type_ids" in wrong_contexts_ids:
                        p_dict["token_type_ids"] = wrong_contexts_ids["token_type_ids"][
                            c_index
                        ]
                    retrieved_hard_negatives.append(p_dict)
                update_dict[prediction["sample_idx"]][
                    "retrieved_hard_negatives"
                ] = retrieved_hard_negatives
            logger.log(f"Adding hard negatives to the dataset.")
            self.dataset.add_fields_to_samples(update_dict)

        return predictions
