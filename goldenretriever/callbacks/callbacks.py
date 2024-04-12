import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import hydra
import torch
from composer import Callback, Event, Logger, State, Time, TimeUnit
from composer.utils import create_interval_scheduler, dist, reproducibility
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from goldenretriever.callbacks.base import NLPTemplateCallback
from goldenretriever.common.log import get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.composer.model import GoldenRetrieverComposerModule
from goldenretriever.data.utils import HardNegativesManager
from goldenretriever.indexers.base import BaseDocumentIndex
from goldenretriever.pytorch.model import GoldenRetriever

program_logger = get_logger(__name__, level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PredictionCallback(Callback):

    def __init__(
        self,
        k: int | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        document_index: BaseDocumentIndex | None = None,
        precision: str | int = 32,
        compute_on_cpu: bool = False,
        force_reindex: bool = True,
        retriever_dir: Path | None = None,
        events: Set[Event] | None = None,
        metric_callbacks: List[DictConfig] | List["NLPTemplateCallback"] | None = None,
        interval: str | int | Time = None,
        dataloader: DataLoader | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.document_index = document_index
        self.precision = precision
        self.compute_on_cpu = compute_on_cpu
        self.force_reindex = force_reindex
        self.retriever_dir = retriever_dir
        self.events = events or {Event.EVAL_END}
        self.dataloader = dataloader

        self.check_interval = create_interval_scheduler(
            interval, include_end_of_training=True
        )
        self.last_generate_batch: Optional[Time] = None

        self.metric_callbacks = metric_callbacks or []
        for i, callback in enumerate(self.metric_callbacks):
            if isinstance(callback, DictConfig):
                self.metric_callbacks[i] = hydra.utils.instantiate(
                    callback, _recursive_=False
                )

    def run_event(self, event: Event, state: State, logger: Logger):
        if event in self.events:
            # if state.get_elapsed_duration(
            # ) is not None and self.check_interval(state, event) and self.last_generate_batch != state.timestamp.batch:
            start = time.time()
            predictions = self(event, state, logger)
            # run the inner callbacks
            # TODO: check if we need rank 0 only
            for callback in self.metric_callbacks:
                callback(
                    state=state,
                    logger=logger,
                    predictions=predictions,
                    callback=self,
                )
            diff = time.time() - start
            program_logger.info(f"{self.__class__.__name__} ran in {diff} seconds.")

    @torch.no_grad()
    def __call__(
        self,
        event: Event,
        state: State,
        logger: Logger,
        *args,
        **kwargs,
    ) -> None:
        """
        Computes the predictions for the dataset.

        Args:
            event (:obj:`Event`):

            state (:obj:`State`):
                The state object.
            logger (:obj:`Logger`):
                The logger object.
        """
        self.last_generate_batch = state.timestamp.batch

        program_logger.info(f"Computing predictions for event `{event}`")
        # get model from state
        composer_model: GoldenRetrieverComposerModule = (
            state.model.module if state.is_model_ddp else state.model
        )
        # Set to evaluation mode and stash the original mode.
        original_mode = composer_model.training
        # device = state.device
        composer_model.eval()
        # dummy forward call needed for FSDP to work consistently
        composer_model.dummy_forward_called = False
        # get the retriever
        retriever: GoldenRetriever = composer_model.model
        # get the current eval dataloader
        dataloader = self.dataloader or state.dataloader
        # get the current eval dataset
        dataset = dataloader.dataset
        # get the tokenizer
        # tokenizer = retriever.question_tokenizer
        passage_tokenizer = dataset.passage_tokenizer

        def collate_fn(x):
            return ModelInputs(
                passage_tokenizer(
                    x,
                    truncation=True,
                    padding=True,
                    max_length=dataset.max_passage_length,
                    return_tensors="pt",
                )
            )

        # check if we need to reindex the passages and
        # also if we need to load the retriever from disk
        if (self.retriever_dir is not None and state.eval_timestamp.epoch == 0) or (
            # self.retriever_dir is not None and event == event.
        ):
            force_reindex = False
        else:
            force_reindex = self.force_reindex

        if (
            not force_reindex
            and self.retriever_dir is not None
            # and stage == RunningStage.TESTING
        ):
            retriever = retriever.from_pretrained(self.retriever_dir)

        # you never know :)
        retriever.eval()
        program_logger.info(
            f"Computing passage embeddings for dataset {state.dataloader_label}"
        )
        retriever.index(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            max_length=dataset.max_passage_length,
            collate_fn=collate_fn,
            precision=self.precision,
            compute_on_cpu=self.compute_on_cpu,
            force_reindex=force_reindex,
        )
        predictions = []
        # start = time.time()
        progress_bar = tqdm(
            dataloader,
            desc=f"Computing predictions for dataset {state.dataloader_label}",
        )
        retriever_outputs = []
        for batch in progress_bar:
            batch = batch.to(state.device._device)
            # get the top-k indices
            retriever_output = retriever.retrieve(
                **batch.questions, k=self.k, precision=self.precision
            )
            retriever_outputs.append((batch, retriever_output))
            progress_bar.update(batch.questions["input_ids"].size(0))

        if dist.get_global_rank() == 0:
            # compute recall at k
            for batch, retriever_output in retriever_outputs:
                for batch_idx, retrieved_samples in enumerate(retriever_output):
                    # get the positive passages
                    gold_passages = batch["positives"][batch_idx]
                    # get the index of the gold passages in the retrieved passages
                    gold_passage_indices = []
                    for passage in gold_passages:
                        try:
                            gold_passage_indices.append(
                                retriever.get_index_from_passage(passage)
                            )
                        except ValueError:
                            logger.warning(
                                f"Passage `{passage}` not found in the index. "
                                "We will skip it, but the results might not reflect the "
                                "actual performance."
                            )
                            pass
                    retrieved_indices = [r.document.id for r in retrieved_samples if r]
                    retrieved_passages = [
                        retriever.get_passage_from_index(i) for i in retrieved_indices
                    ]
                    retrieved_scores = [r.score for r in retrieved_samples]
                    # correct predictions are the passages that are in the top-k and are gold
                    correct_indices = set(gold_passage_indices) & set(retrieved_indices)
                    # wrong predictions are the passages that are in the top-k and are not gold
                    wrong_indices = set(retrieved_indices) - set(gold_passage_indices)
                    # add the predictions to the list
                    prediction_output = dict(
                        sample_idx=batch.sample_idx[batch_idx],
                        gold=gold_passages,
                        predictions=retrieved_passages,
                        scores=retrieved_scores,
                        correct=[
                            retriever.get_passage_from_index(i) for i in correct_indices
                        ],
                        wrong=[
                            retriever.get_passage_from_index(i) for i in wrong_indices
                        ],
                    )
                    predictions.append(prediction_output)

        composer_model.train(mode=original_mode)
        return predictions


class HardNegativeMiningCallback(PredictionCallback):

    def __init__(
        self,
        k: int | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        document_index: BaseDocumentIndex | None = None,
        precision: str | int = 32,
        compute_on_cpu: bool = False,
        force_reindex: bool = True,
        retriever_dir: Path | None = None,
        events: Set[Event] | None = None,
        metric_callbacks: List[DictConfig] | List["NLPTemplateCallback"] | None = None,
        interval: str | int | Time = None,
        dataloader: DataLoader | None = None,
        metrics_to_monitor: str = None,
        threshold: float = 0.8,
        max_negatives: int = 5,
        add_with_probability: float = 1.0,
        *args,
        **kwargs,
    ):
        if dataloader is None:
            raise ValueError("For hard negative mining, a dataloader must be provided.")
        # overwrite the events if not provided
        events = events or {Event.EVAL_AFTER_ALL}
        super().__init__(
            k=k,
            batch_size=batch_size,
            num_workers=num_workers,
            document_index=document_index,
            precision=precision,
            compute_on_cpu=compute_on_cpu,
            force_reindex=force_reindex,
            retriever_dir=retriever_dir,
            events=events,
            metric_callbacks=metric_callbacks,
            interval=interval,
            dataloader=dataloader,
            *args,
            **kwargs,
        )
        self.metrics_to_monitor = f"metrics/{metrics_to_monitor}"
        self.threshold = threshold
        self.max_negatives = max_negatives
        self.add_with_probability = add_with_probability

    @torch.no_grad()
    def __call__(
        self,
        event: Event,
        state: State,
        logger: Logger,
        *args,
        **kwargs,
    ) -> None:
        if event not in self.events:
            return

        if self.metrics_to_monitor not in state.eval_metrics:
            logger.warning(
                f"Metric `{self.metrics_to_monitor}` not found in trainer.logged_metrics. "
                f"Available metrics: {state.eval_metrics.keys()}"
            )
            return {}

        if state.eval_metrics[self.metrics_to_monitor] < self.threshold:
            return {}

        program_logger.info(f"Computing hard negatives predictions for event `{event}`")
        # get model from state
        composer_model: GoldenRetrieverComposerModule = (
            state.model.module if state.is_model_ddp else state.model
        )
        # Set to evaluation mode and stash the original mode.
        original_mode = composer_model.training
        # device = state.device
        composer_model.eval()
        # dummy forward call needed for FSDP to work consistently
        composer_model.dummy_forward_called = False
        # get the retriever
        retriever = composer_model.model
        # get the current train dataset
        dataset = self.dataloader.dataset
        # save the current epoch of the training
        current_train_epoch = state.timestamp.epoch
        # get the tokenizer
        passage_tokenizer = dataset.passage_tokenizer
        # restore the state of the dataset
        self.dataloader.dataset.load_state_dict(
            state.train_dataloader.dataset.state_dict(
                int(state.timestamp.sample_in_epoch.value), True
            )
        )

        # get the steps before the next evaluation
        # this information is stored in the evaluators
        # get the maximum interval step from the evaluators
        n_samples_to_augment = [
            evaluator.actual_eval_interval for evaluator in state.evaluators
        ]
        # get the unit of the interval: if any has "ep" we annotate all the dataset
        if any(n.unit == TimeUnit.EPOCH for n in n_samples_to_augment):
            n_samples_to_augment = None
        else:
            # cast to int and get the maximum value
            n_samples_to_augment = max(
                [int(n.value) for n in n_samples_to_augment if n is not None]
            )

        def collate_fn(x):
            return ModelInputs(
                passage_tokenizer(
                    x,
                    truncation=True,
                    padding=True,
                    max_length=dataset.max_passage_length,
                    return_tensors="pt",
                )
            )

        # check if we need to reindex the passages and
        # also if we need to load the retriever from disk
        if (self.retriever_dir is not None and state.eval_timestamp.epoch == 0) or (
            # self.retriever_dir is not None and event == event.
        ):
            force_reindex = False
        else:
            force_reindex = self.force_reindex

        if (
            not force_reindex
            and self.retriever_dir is not None
            # and stage == RunningStage.TESTING
        ):
            retriever = retriever.from_pretrained(self.retriever_dir)
        # you never know :)
        retriever.eval()

        program_logger.info(
            f"Computing passage embeddings for dataset {state.dataloader_label}"
        )
        # retriever.index(
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     max_length=dataset.max_passage_length,
        #     collate_fn=collate_fn,
        #     precision=self.precision,
        #     compute_on_cpu=False,
        #     force_reindex=force_reindex,
        # )
        predictions = []

        reproducibility.seed_all(state.seed)

        # start = time.time()
        progress_bar = tqdm(
            self.dataloader,
            initial=0,
            total=n_samples_to_augment or len(self.dataloader),
            desc=f"Computing predictions for dataset {state.dataloader_label}",
        )
        batch_augmented = 0
        for batch in progress_bar:
            batch = batch.to(state.device._device)
            # get the top-k indices
            retriever_output = retriever.retrieve(
                **batch.questions, k=self.k, precision=self.precision
            )
            # compute recall at k
            for batch_idx, retrieved_samples in enumerate(retriever_output):
                # get the positive passages
                gold_passages = batch["positives"][batch_idx]
                # get the index of the gold passages in the retrieved passages
                gold_passage_indices = []
                for passage in gold_passages:
                    try:
                        gold_passage_indices.append(
                            retriever.get_index_from_passage(passage)
                        )
                    except ValueError:
                        logger.warning(
                            f"Passage `{passage}` not found in the index. "
                            "We will skip it, but the results might not reflect the "
                            "actual performance."
                        )
                        pass
                retrieved_indices = [r.document.id for r in retrieved_samples if r]
                retrieved_passages = [
                    retriever.get_passage_from_index(i) for i in retrieved_indices
                ]
                retrieved_scores = [r.score for r in retrieved_samples]
                # correct predictions are the passages that are in the top-k and are gold
                correct_indices = set(gold_passage_indices) & set(retrieved_indices)
                # wrong predictions are the passages that are in the top-k and are not gold
                wrong_indices = set(retrieved_indices) - set(gold_passage_indices)
                # add the predictions to the list
                prediction_output = dict(
                    sample_idx=batch.sample_idx[batch_idx],
                    gold=gold_passages,
                    predictions=retrieved_passages,
                    scores=retrieved_scores,
                    correct=[
                        retriever.get_passage_from_index(i) for i in correct_indices
                    ],
                    wrong=[retriever.get_passage_from_index(i) for i in wrong_indices],
                )
                predictions.append(prediction_output)
            progress_bar.update(batch.questions["input_ids"].size(0))
            batch_augmented += 1
            if (
                n_samples_to_augment is not None
                and batch_augmented >= n_samples_to_augment
            ):
                program_logger.info(
                    f"Augmented next iteration batches ({batch_augmented}). "
                    "Stopping the hard negative mining."
                )
                break

        # dataset.hn_manager = None
        program_logger.info(f"Computing hard negatives for epoch {current_train_epoch}")
        # predictions is a dict with the dataloader index as key and the predictions as value
        # since we only have one dataloader, we can get the predictions directly
        # predictions = list(predictions.values())[0]
        # store the predictions in a dictionary for faster access based on the sample index
        hard_negatives_list = {}
        for prediction in tqdm(predictions, desc="Collecting hard negatives"):
            if random.random() < 1 - self.add_with_probability:
                continue
            top_k_passages = prediction["predictions"]
            gold_passages = prediction["gold"]
            # get the ids of the max_negatives wrong passages with the highest similarity
            wrong_passages = [
                passage_id
                for passage_id in top_k_passages
                if passage_id not in gold_passages
            ][: self.max_negatives]
            hard_negatives_list[prediction["sample_idx"]] = wrong_passages

        # HardNegativesManager is a singleton, so we need
        # to reset it before adding new hard negatives
        hn_manager = HardNegativesManager(
            tokenizer=passage_tokenizer,
            max_length=dataset.max_passage_length,
        )
        hn_manager.reset()
        hn_manager.add(hard_negatives_list.keys(), hard_negatives_list.values())
        hn_manager.tokenize()

        composer_model.train(mode=original_mode)
        return predictions


class RecallAtKEvaluationCallback(NLPTemplateCallback):
    """
    Computes the recall at k for the predictions. Recall at k is computed as the number of
    correct predictions in the top k predictions divided by the total number of correct
    predictions.

    Args:
        k (`int`):
            The number of predictions to consider.
        prefix (`str`, `optional`):
            The prefix to add to the metrics.
        verbose (`bool`, `optional`, defaults to `False`):
            Whether to log the metrics.
        prog_bar (`bool`, `optional`, defaults to `True`):
            Whether to log the metrics to the progress bar.
    """

    def __init__(
        self,
        k: int = 100,
        prefix: str | None = None,
        verbose: bool = True,
        prog_bar: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.prefix = prefix
        self.verbose = verbose
        self.prog_bar = prog_bar

    @torch.no_grad()
    def __call__(
        self,
        state: State,
        logger: Logger,
        predictions: Dict,
        *args,
        **kwargs,
    ) -> dict:
        """
        Computes the recall at k for the predictions.

        Args:
            trainer (:obj:`lightning.trainer.trainer.Trainer`):
                The trainer object.
            pl_module (:obj:`lightning.core.lightning.LightningModule`):
                The lightning module.
            predictions (:obj:`Dict`):
                The predictions.

        Returns:
            :obj:`Dict`: The computed metrics.
        """
        # metrics to return
        metrics = {}

        # stage = trainer.state.stage
        # if stage not in DEFAULT_STAGES:
        #     raise ValueError(
        #         f"Stage {stage} not supported, only `validate` and `test` are supported."
        #     )

        dataloader_label = state.dataloader_label
        # for dataloader_idx, samples in predictions.items():
        hits, total = 0, 0
        for sample in predictions:
            # for sample in samples:
            # compute the recall at k
            # cut the predictions to the first k elements
            predictions = sample["predictions"][: self.k]
            hits += len(set(predictions) & set(sample["gold"]))
            total += len(set(sample["gold"]))

        # compute the mean recall at k
        recall_at_k = hits / total
        # metrics[f"recall@{self.k}_{dataloader_label}"] = recall_at_k
        # metrics[f"recall@{self.k}"] = sum(metrics.values()) / len(metrics)

        # if self.prefix is not None:
        #     metrics = {f"metrics/recall/{self.prefix}_{k}": v for k, v in metrics.items()}
        # else:
        # metrics = {f"metrics/{dataloader_label}/{k}": v for k, v in metrics.items()}
        # pl_module.log_dict(
        #     metrics, on_step=False, on_epoch=True, prog_bar=self.prog_bar
        # )
        logger.log_metrics({f"metrics/{dataloader_label}/recall@{self.k}": recall_at_k})
        state.eval_metrics[f"metrics/{dataloader_label}/recall@{self.k}"] = recall_at_k
        state.eval_metrics[f"metrics/recall@{self.k}"] = recall_at_k
        if self.verbose:
            program_logger.info(
                f"metrics/{dataloader_label}/recall@{self.k}: {recall_at_k}"
            )

        return metrics
