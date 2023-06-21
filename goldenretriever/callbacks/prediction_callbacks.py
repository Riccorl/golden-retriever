import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Set, Union

import psutil
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from tqdm import tqdm

from goldenretriever.callbacks.base import PredictionCallback
from goldenretriever.common.log import get_console_logger, get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.data.datasets import BaseDataset
from goldenretriever.models.model import GoldenRetriever

console_logger = get_console_logger()
logger = get_logger(__name__, level=logging.INFO)


class GoldenRetrieverPredictionCallback(PredictionCallback):
    def __init__(
        self,
        k: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        use_faiss: bool = False,
        move_index_to_cpu: bool = True,
        precision: Union[str, int] = 32,
        index_precision: Union[str, int] = 32,
        force_reindex: bool = True,
        retriever_dir: Optional[Path] = None,
        stages: Optional[Set[Union[str, RunningStage]]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        dataloader: Optional[DataLoader] = None,
        *args,
        **kwargs,
    ):
        super().__init__(batch_size, stages, other_callbacks, dataset, dataloader)
        self.k = k
        self.num_workers = num_workers
        self.use_faiss = use_faiss
        self.move_index_to_cpu = move_index_to_cpu
        self.precision = precision
        self.index_precision = index_precision
        self.force_reindex = force_reindex
        self.retriever_dir = retriever_dir

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        datasets: Optional[
            Union[DictConfig, BaseDataset, List[DictConfig], List[BaseDataset]]
        ] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        *args,
        **kwargs,
    ) -> dict:
        stage = trainer.state.stage
        logger.info(f"Computing predictions for stage {stage.value}")
        if stage not in self.stages:
            raise ValueError(
                f"Stage {stage} not supported, only {self.stages} are supported"
            )

        # get the tokenizer
        tokenizer = trainer.datamodule.tokenizer

        # if datasets is not None or dataloaders is not None:
        #     self.datasets = datasets
        #     self.dataloaders = dataloaders

        self.datasets, self.dataloaders = self._get_datasets_and_dataloaders(
            datasets,
            dataloaders,
            trainer,
            dataloader_kwargs=dict(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
            ),
            collate_fn_kwargs=dict(tokenizer=tokenizer),
        )

        # set the model to eval mode
        pl_module.eval()
        # get the retriever
        retriever: GoldenRetriever = pl_module.model

        # here we will store the samples with predictions for each dataloader
        dataloader_predictions = {}
        # compute the context embeddings index for each dataloader
        for dataloader_idx, dataloader in enumerate(self.dataloaders):
            current_dataset: BaseDataset = self.datasets[dataloader_idx]
            logger.info(
                f"Computing context embeddings for dataset {current_dataset.name}"
            )
            contexts = self._get_contexts_dataloader(current_dataset, trainer)

            collate_fn = lambda x: ModelInputs(
                tokenizer(
                    x,
                    truncation=True,
                    padding=True,
                    max_length=current_dataset.max_context_length,
                    return_tensors="pt",
                )
            )

            # check if we need to reindex the contexts and
            # also if we need to load the retriever from disk
            if (self.retriever_dir is not None and trainer.current_epoch == 0) or (
                self.retriever_dir is not None and stage == RunningStage.TESTING
            ):
                force_reindex = False
            else:
                force_reindex = self.force_reindex

            if (
                not force_reindex
                and self.retriever_dir is not None
                and stage == RunningStage.TESTING
            ):
                retriever = retriever.from_pretrained(self.retriever_dir)
                # set the retriever to eval mode if we are loading it from disk

            # you never know :)
            retriever.eval()

            retriever.index(
                contexts,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                force_reindex=force_reindex,
                use_faiss=self.use_faiss,
                move_index_to_cpu=self.move_index_to_cpu,
                precision=self.precision,
                index_precision=self.index_precision,
            )

            pl_module_original_device = pl_module.device
            if (
                not self.use_faiss
                and self.move_index_to_cpu
                and pl_module.device.type == "cuda"
            ):
                pl_module.to("cpu")

            # now compute the question embeddings and compute the top-k accuracy
            # logger.info(f"Computing predictions for dataset {current_dataset.name}")
            predictions = []
            start = time.time()
            for batch in tqdm(
                dataloader,
                desc=f"Computing predictions for dataset {current_dataset.name}",
            ):
                batch = batch.to(pl_module.device)
                # get the top-k indices
                retriever_output = retriever.retrieve(
                    **batch.questions, k=self.k, precision=self.precision
                )
                # compute recall at k
                for batch_idx, retrieved_samples in enumerate(retriever_output):
                    # get the positive contexts
                    gold_contexts = batch.positives[batch_idx]
                    # get the index of the gold contexts in the retrieved contexts
                    gold_context_indices = [
                        retriever.get_index_from_context(context)
                        for context in gold_contexts
                    ]
                    retrieved_indices = [r.index for r in retrieved_samples]
                    retrieved_contexts = [r.label for r in retrieved_samples]
                    retrieved_scores = [r.score for r in retrieved_samples]
                    # correct predictions are the contexts that are in the top-k and are gold
                    correct_indices = set(gold_context_indices) & set(retrieved_indices)
                    # wrong predictions are the contexts that are in the top-k and are not gold
                    wrong_indices = set(retrieved_indices) - set(gold_context_indices)
                    # add the predictions to the list
                    prediction_output = dict(
                        sample_idx=batch.sample_idx[batch_idx],
                        gold=gold_contexts,
                        predictions=retrieved_contexts,
                        scores=retrieved_scores,
                        correct=[
                            retriever.get_context_from_index(i) for i in correct_indices
                        ],
                        wrong=[
                            retriever.get_context_from_index(i) for i in wrong_indices
                        ],
                    )
                    # if "id" in batch:
                    #     prediction_output["id"] = batch.id[batch_idx]
                    predictions.append(prediction_output)
            end = time.time()
            logger.info(f"Time to retrieve: {str(end - start)}")

            dataloader_predictions[dataloader_idx] = predictions

            if pl_module_original_device != pl_module.device:
                pl_module.to(pl_module_original_device)

        # return the predictions
        return dataloader_predictions

    @staticmethod
    def _get_contexts_dataloader(dataset, trainer):
        if dataset.contexts is None:
            logger.info(
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


class NegativeAugmentationCallback(GoldenRetrieverPredictionCallback):
    def __init__(
        self,
        k: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        use_faiss: bool = False,
        move_index_to_cpu: bool = False,
        force_reindex: bool = False,
        retriever_dir: Optional[Path] = None,
        stages: Set[Union[str, RunningStage]] = None,
        other_callbacks: Optional[List[DictConfig]] = None,
        dataset: Optional[Union[DictConfig, BaseDataset]] = None,
        metrics_to_monitor: List[str] = None,
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
            retriever_dir=retriever_dir,
            stages=stages,
            other_callbacks=other_callbacks,
            dataset=dataset,
            *args,
            **kwargs,
        )
        if metrics_to_monitor is None:
            metrics_to_monitor = ["val_loss"]
        self.metrics_to_monitor = metrics_to_monitor
        self.threshold = threshold
        self.max_negatives = max_negatives
        self.refresh_every_n_epochs = refresh_every_n_epochs

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> dict:
        stage = trainer.state.stage
        if stage not in self.stages:
            return {}

        if self.metrics_to_monitor not in trainer.logged_metrics:
            raise ValueError(
                f"Metric {self.metric_to_monitor} not found in trainer.logged_metrics"
                f"Available metrics: {trainer.logged_metrics.keys()}"
            )
        if trainer.logged_metrics[self.metrics_to_monitor] < self.threshold:
            return {}

        if trainer.current_epoch % self.refresh_every_n_epochs != 0:
            return {}

        # if all(
        #     [
        #         trainer.logged_metrics.get(metric) is None
        #         for metric in self.metrics_to_monitor
        #     ]
        # ):
        #     raise ValueError(
        #         f"No metric from {self.metrics_to_monitor} not found in trainer.logged_metrics"
        #         f"Available metrics: {trainer.logged_metrics.keys()}"
        #     )

        # if all(
        #     [
        #         trainer.logged_metrics.get(metric) < self.threshold
        #         for metric in self.metrics_to_monitor
        #         if trainer.logged_metrics.get(metric) is not None
        #     ]
        # ):
        #     return {}

        if trainer.current_epoch % self.refresh_every_n_epochs != 0:
            return {}

        logger.info(
            f"At least one metric from {self.metrics_to_monitor} is above threshold "
            f"{self.threshold}. Computing hard negatives."
        )

        predictions = super().__call__(
            trainer,
            pl_module,
            datasets=trainer.datamodule.train_dataset,
            dataloaders=trainer.datamodule.train_dataloader(),
            *args,
            **kwargs,
        )
        logger.info(f"Computing hard negatives for epoch {trainer.current_epoch}")
        # predictions is a dict with the dataloader index as key and the predictions as value
        # since we only have one dataloader, we can get the predictions directly
        predictions = list(predictions.values())[0]
        # store the predictions in a dictionary for faster access based on the sample index
        # hard_negatives_dict = {"sample_idx": [], "contexts": []}
        hard_negatives_dict = []
        for prediction in tqdm(predictions, desc="Collecting hard negatives"):
            top_k_contexts = prediction["predictions"]
            gold_contexts = prediction["gold"]
            # get the ids of the max_negatives wrong contexts with the highest similarity
            wrong_contexts = [
                context_id
                for context_id in top_k_contexts
                if context_id not in gold_contexts
            ][: self.max_negatives]
            # hard_negatives_dict[prediction["sample_idx"]] = wrong_contexts
            # hard_negatives_dict["sample_idx"].append(prediction["sample_idx"])
            # hard_negatives_dict["contexts"].append(wrong_contexts)
            hard_negatives_dict.append(
                dict(sample_idx=prediction["sample_idx"], contexts=wrong_contexts)
            )

        # convert the dictionary to a dataset for easy multiprocessing
        # hard_negatives_ds = Dataset.from_dict(hard_negatives_dict)
        # dump in a temporary file and load it again into a dataset
        # this is necessary because the dataset cannot load in-memory data
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / "hard_negatives.jsonl"
            with open(tmp_file, "w") as f:
                for sample in hard_negatives_dict:
                    f.write(json.dumps(sample) + "\n")
            # clean hard_negatives_dict
            hard_negatives_dict = None
            # load the dataset from the temporary file
            hard_negatives_ds = load_dataset(
                "json", data_files=[str(tmp_file)], split="train"
            )

        def map_to_hn_dataset(sample, tokenizer, max_length):
            contexts = sample["contexts"]
            context_ids = tokenizer(
                contexts,
                max_length=max_length,
                truncation=True,
            )
            inverted_context_ids = []
            for index in range(len(contexts)):
                if "token_type_ids" in context_ids:
                    inverted_context_ids.append(
                        dict(
                            input_ids=context_ids["input_ids"][index],
                            attention_mask=context_ids["attention_mask"][index],
                            token_type_ids=context_ids["token_type_ids"][index],
                        )
                    )
                else:
                    inverted_context_ids.append(
                        dict(
                            input_ids=context_ids["input_ids"][index],
                            attention_mask=context_ids["attention_mask"][index],
                        )
                    )
            sample["hard_negatives"] = inverted_context_ids

        hard_negatives_ds = hard_negatives_ds.map(
            map_to_hn_dataset,
            fn_kwargs=dict(
                tokenizer=trainer.datamodule.tokenizer,
                max_length=trainer.datamodule.train_dataset.max_context_length,
            ),
            num_proc=psutil.cpu_count(),
            desc="Tokenizing hard negatives",
        )
        trainer.datamodule.train_dataset.hard_negatives_dict

        # wrong_contexts_ids = trainer.datamodule.tokenizer(
        #     wrong_contexts,
        #     max_length=trainer.datamodule.train_dataset.max_context_length,
        #     truncation=True,
        # )
        # hard_negatives_per_sample = []
        # for idx, input_ids in enumerate(wrong_contexts_ids["input_ids"]):
        #     if "token_type_ids" in wrong_contexts_ids:
        #         hn_sample_dict = {
        #             "input_ids": input_ids,
        #             "attention_mask": wrong_contexts_ids["attention_mask"][idx],
        #             "token_type_ids": wrong_contexts_ids["token_type_ids"][idx],
        #         }
        #     else:
        #         hn_sample_dict = {
        #             "input_ids": input_ids,
        #             "attention_mask": wrong_contexts_ids["attention_mask"][idx],
        #         }
        #     hard_negatives_per_sample.append(hn_sample_dict)

        # for context_index in range(len(wrong_contexts)):
        #     p_dict = {
        #         "input_ids": wrong_contexts_ids["input_ids"][context_index],
        #         "attention_mask": wrong_contexts_ids["attention_mask"][context_index],
        #     }
        #     if "token_type_ids" in wrong_contexts_ids:
        #         p_dict["token_type_ids"] = wrong_contexts_ids["token_type_ids"][
        #             context_index
        #         ]
        #     hard_negs.append(p_dict)
        # hard_negatives_dict[prediction["sample_idx"]] = hard_negatives_per_sample

        # tokenization for the hard negatives
        # tokenized_hard_negatives = {}
        # for sample_idx, wrong_contexts in tqdm(
        #     hard_negatives_dict.items(), desc="Tokenizing hard negatives"
        # ):
        #     wrong_contexts_ids = trainer.datamodule.tokenizer(
        #         wrong_contexts,
        #         max_length=trainer.datamodule.train_dataset.max_context_length,
        #         truncation=True,
        #     )
        #     hard_negs = []
        #     for c_index in range(len(wrong_contexts)):
        #         p_dict = {
        #             "input_ids": wrong_contexts_ids["input_ids"][c_index],
        #             "attention_mask": wrong_contexts_ids["attention_mask"][c_index],
        #         }
        #         if "token_type_ids" in wrong_contexts_ids:
        #             p_dict["token_type_ids"] = wrong_contexts_ids["token_type_ids"][
        #                 c_index
        #             ]
        #         hard_negs.append(p_dict)
        #     tokenized_hard_negatives[sample_idx] = hard_negs

        # # # update the dataset with the hard negatives
        # # trainer.datamodule.train_dataset.hard_negatives_dict = hard_negatives_dict
        # trainer.datamodule.train_dataset.hard_negatives_dict = tokenized_hard_negatives

        # retrieved_hard_negatives.append(wrong_contexts)
        # wrong_contexts_ids = trainer.datamodule.tokenizer(
        #     wrong_contexts,
        #     max_length=trainer.datamodule.train_dataset.max_context_length,
        #     truncation=True,
        # )
        # retrieved_hard_negatives = []
        # for c_index in range(len(wrong_contexts)):
        #     p_dict = {
        #         "input_ids": wrong_contexts_ids["input_ids"][c_index],
        #         "attention_mask": wrong_contexts_ids["attention_mask"][c_index],
        #     }
        #     if "token_type_ids" in wrong_contexts_ids:
        #         p_dict["token_type_ids"] = wrong_contexts_ids["token_type_ids"][
        #             c_index
        #         ]
        #     retrieved_hard_negatives.append(p_dict)
        # update_dict[prediction["sample_idx"]][
        #     "retrieved_hard_negatives"
        # ] = retrieved_hard_negatives

        # trainer.datamodule.train_dataset.hard_negatives_dict = tokenized_hard_negatives

        # def _add_hard_negatives_to_sample(
        #     sample, hard_negatives, tokenizer, max_context_length
        # ):
        #     wrong_contexts = hard_negatives[sample["sample_idx"]]
        #     wrong_contexts_ids = tokenizer(
        #         wrong_contexts,
        #         max_length=max_context_length,
        #         truncation=True,
        #     )
        #     hard_negatives_per_sample = []
        #     for c_index in range(len(wrong_contexts)):
        #         sample_hn = {
        #             "input_ids": wrong_contexts_ids["input_ids"][c_index],
        #             "attention_mask": wrong_contexts_ids["attention_mask"][c_index],
        #         }
        #         if "token_type_ids" in wrong_contexts_ids:
        #             sample_hn["token_type_ids"] = wrong_contexts_ids[
        #                 "token_type_ids"
        #             ][c_index]
        #         hard_negatives_per_sample.append(sample_hn)
        #     sample["retrieved_hard_negatives"] = hard_negatives_per_sample
        #     return sample

        # if isinstance(trainer.datamodule.train_dataset.data, IterableDataset):
        #     map_kwargs = {
        #         "function": _add_hard_negatives_to_sample,
        #         "fn_kwargs": {"hard_negatives": hard_negatives_dict},
        #     }
        # else:
        #     map_kwargs = {
        #         "function": _add_hard_negatives_to_sample,
        #         "fn_kwargs": {
        #             "hard_negatives": hard_negatives_dict,
        #             "tokenizer": trainer.datamodule.tokenizer,
        #             "max_context_length": trainer.datamodule.train_dataset.max_context_length,
        #         },
        #         # "num_proc": psutil.cpu_count(),
        #         "desc": "Adding hard negatives to train dataset",
        #     }
        # trainer.datamodule.train_dataset.data = (
        #     trainer.datamodule.train_dataset.data.map(**map_kwargs)
        # )

        # if train_dataset.data is a IterableDataset, we should "activate" the map function
        # by iterating over the dataset
        # if isinstance(trainer.datamodule.train_dataset.data, IterableDataset):
        #     for _ in tqdm(
        #         trainer.datamodule.train_dataset.data,
        #         desc="Adding hard negatives to train dataset",
        #     ):
        #         pass

        # trainer.datamodule.train_dataset.data_iterator = iter(
        #     trainer.datamodule.train_dataset.data
        # )
        # trainer.datamodule.train_dataset.add_fields_to_samples(update_dict)

        # normalize predictions as in the original GoldenRetrieverPredictionCallback
        predictions = {0: predictions}
        return predictions