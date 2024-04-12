import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import composer
import composer.callbacks
import composer.optim
import hydra
import lightning as pl
import omegaconf
import torch
from composer import (
    Algorithm,
    ComposerModel,
    DataSpec,
    Evaluator,
    Event,
    State,
    Time,
    TimeUnit,
)
from composer import Trainer as ComposerTrainer
from composer.algorithms import GradientClipping
from composer.callbacks import (
    CheckpointSaver,
    EarlyStopper,
    LRMonitor,
    MemoryMonitor,
    SpeedMonitor,
)
from composer.loggers import WandBLogger
from composer.optim.scheduler import LinearScheduler
from composer.utils import dist, get_device, reproducibility
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from goldenretriever.callbacks.base import NLPTemplateCallback
from goldenretriever.callbacks.callbacks import (
    HardNegativeMiningCallback,
    PredictionCallback,
)
from goldenretriever.callbacks.callbacks import (
    RecallAtKEvaluationCallback as ComposerRecallAtKEvaluationCallback,
)
from goldenretriever.callbacks.checkpoint_saver import MetricCheckpointSaver
from goldenretriever.common.from_config import FromConfig
from goldenretriever.common.log import get_logger
from goldenretriever.common.utils import to_config
from goldenretriever.composer.algorithms import HardNegativeAlgorithm
from goldenretriever.composer.model import GoldenRetrieverComposerModule
from goldenretriever.data.datasets import (
    GoldenRetrieverCollator,
    GoldenRetrieverStreamingDataset,
)
from goldenretriever.pytorch.loss import MultiLabelNCELoss
from goldenretriever.pytorch.model import GoldenRetriever
from goldenretriever.pytorch.optim import RAdamW
from goldenretriever.trainer import COMPOSER_PRECISION_INPUT_STR_ALIAS_CONVERSION
from goldenretriever.trainer.evaluator import GoldenRetrieverEvaluator

logger = get_logger()


class Trainer(FromConfig):
    def __init__(
        self,
        retriever: GoldenRetriever,
        train_dataset: str | GoldenRetrieverStreamingDataset | None = None,
        train_batch_size: int = 32,
        train_dataset_kwargs: dict | None = None,
        val_dataset: (
            str
            | GoldenRetrieverStreamingDataset
            | list[str]
            | list[GoldenRetrieverStreamingDataset]
            | None
        ) = None,
        val_batch_size: int = 32,
        val_dataset_kwargs: dict | list[dict] | None = None,
        test_dataset: (
            str
            | GoldenRetrieverStreamingDataset
            | list[str]
            | list[GoldenRetrieverStreamingDataset]
            | None
        ) = None,
        test_batch_size: int = 32,
        test_dataset_kwargs: dict | list[dict] | None = None,
        num_workers: int = 4,
        optimizer: torch.optim.Optimizer = RAdamW,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        lr_scheduler: composer.optim.scheduler.ComposerScheduler = LinearScheduler,
        lr_kwargs: dict | None = None,
        loss: torch.nn.Module = MultiLabelNCELoss,
        callbacks: list | None = None,
        device: str = "gpu",
        precision: int | str = "amp_fp16",
        gradient_clip_norm: float = 0.1,
        max_duration: str | Time | TimeUnit = "1ba",
        eval_interval: int | str | Time | Callable[[State, Event], bool] = "1ba",
        device_train_microbatch_size: int | str | None = None,
        device_eval_microbatch_size: int | str | None = None,
        step_schedulers_every_batch: bool = True,
        deterministic: bool = True,
        fast_dev_run: bool = False,
        resume_from_checkpoint_path: str | os.PathLike | None = None,
        deepspeed_config: dict | None = None,
        fsdp_config: dict | None = None,
        dist_timeout: float = 300.0,
        composer_trainer_kwargs: dict | None = None,
        # eval parameters
        metric_to_monitor: str = "recall@{top_k}",
        monitor_mode: str = "max",
        top_k: int | List[int] = 100,
        # early stopping parameters
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_kwargs: dict | None = None,
        # wandb logger parameters
        log_to_wandb: bool = True,
        wandb_entity: str | None = None,
        wandb_experiment_name: str | None = None,
        wandb_project_name: str = "golden-retriever",
        wandb_save_dir: str | os.PathLike = "./",  # TODO: i don't like this default
        wandb_log_artifacts: bool = False,
        wandb_online_mode: bool = False,
        # wandb_watch: str = "all",
        wandb_rank_zero_only: bool = True,
        wandb_kwargs: dict | None = None,
        # checkpoint parameters
        model_checkpointing: bool = True,
        checkpoint_dir: str | os.PathLike = "checkpoints",
        checkpoint_filename: str | os.PathLike | None = None,
        save_top_k: int = 1,
        save_last: bool = False,
        checkpoint_kwargs: dict | None = None,
        # prediction callback parameters
        prediction_batch_size: int = 128,
        # hard negatives callback parameters
        max_hard_negatives_to_mine: int = 15,
        hard_negatives_threshold: float = 0.0,
        metrics_to_monitor_for_hard_negatives: str | None = None,
        mine_hard_negatives_with_probability: float = 1.0,
        # other parameters
        progress_bar: bool = True,
        log_to_console: bool = False,
        seed: int = 42,
        float32_matmul_precision: str = "medium",
        **kwargs,
    ):
        dist.initialize_dist(get_device(None), timeout=dist_timeout)

        # put all the parameters in the class
        self.retriever = retriever
        # datasets
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_dataset_kwargs = train_dataset_kwargs or {}
        self.val_dataset = val_dataset
        self.val_batch_size = val_batch_size
        self.val_dataset_kwargs = val_dataset_kwargs or {}
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size
        self.test_dataset_kwargs = test_dataset_kwargs or {}
        self.num_workers = num_workers
        # trainer parameters
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_kwargs = lr_kwargs or {}
        # self.num_warmup_steps = num_warmup_steps
        self.loss = loss
        self.callbacks = callbacks
        self.device = device
        self.precision = precision
        self.gradient_clip_norm = gradient_clip_norm
        self.max_duration = max_duration
        # self.max_steps = max_steps
        # self.max_epochs = max_epochs
        self.eval_interval = eval_interval
        self.device_train_microbatch_size = (
            device_train_microbatch_size or train_batch_size
        )
        self.device_eval_microbatch_size = device_eval_microbatch_size or val_batch_size
        self.step_schedulers_every_batch = step_schedulers_every_batch
        self.deterministic = deterministic
        self.fast_dev_run = fast_dev_run
        self.resume_from_checkpoint_path = resume_from_checkpoint_path
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        self.dist_timeout = dist_timeout
        self.composer_trainer_kwargs = composer_trainer_kwargs or {}
        # eval parameters
        self.metric_to_monitor = metric_to_monitor
        self.monitor_mode = monitor_mode
        self.top_k = top_k
        # early stopping parameters
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_kwargs = early_stopping_kwargs
        # wandb logger parameters
        self.log_to_wandb = log_to_wandb
        self.wandb_entity = wandb_entity
        self.wandb_experiment_name = wandb_experiment_name
        self.wandb_project_name = wandb_project_name
        self.wandb_save_dir = wandb_save_dir
        self.wandb_log_artifacts = wandb_log_artifacts
        self.wandb_online_mode = wandb_online_mode
        self.wandb_rank_zero_only = wandb_rank_zero_only
        # self.wandb_watch = wandb_watch
        self.wandb_kwargs = wandb_kwargs
        # checkpoint parameters
        self.model_checkpointing = model_checkpointing
        # merge checkpoint dir with wandb project name and experiment name if they are provided
        if (
            self.wandb_project_name is not None
            and self.wandb_experiment_name is not None
        ):
            self.checkpoint_dir = (
                Path(checkpoint_dir)
                / self.wandb_project_name
                / self.wandb_experiment_name
            )
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        # if the dir is relative, make it absolute to the current working directory
        if not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = Path.cwd() / self.checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.checkpoint_kwargs = checkpoint_kwargs
        # prediction callback parameters
        self.prediction_batch_size = prediction_batch_size
        # hard negatives callback parameters
        self.max_hard_negatives_to_mine = max_hard_negatives_to_mine
        self.hard_negatives_threshold = hard_negatives_threshold
        self.metrics_to_monitor_for_hard_negatives = (
            metrics_to_monitor_for_hard_negatives
        )
        self.mine_hard_negatives_with_probability = mine_hard_negatives_with_probability
        # other parameters
        self.progress_bar = progress_bar
        self.log_to_console = log_to_console
        self.seed = seed
        self.float32_matmul_precision = float32_matmul_precision

        # reproducibility
        if self.deterministic:
            reproducibility.configure_deterministic_mode()
        reproducibility.seed_all(self.seed)
        # set the precision of matmul operations
        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        # dataset and dataloader declaration
        (
            self.train_dataset,
            self.val_dataset,
            self.train_dataloader,
            self.val_dataloader,
        ) = self.configure_dataset_and_dataloader()

        # evaluator declaration
        self.evaluators = self.configure_evaluators()

        # optimizer declaration
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # composer module declaration
        self.composer_module = self.configure_composer_module()

        # logger and experiment declaration
        wandb_args = dict(
            project=self.wandb_project_name,
            name=self.wandb_experiment_name,
            entity=self.wandb_entity,
            log_artifacts=self.wandb_log_artifacts,
            rank_zero_only=self.wandb_rank_zero_only,
            online=self.wandb_online_mode,
        )
        wandb_init_kwargs = {
            "save_dir": self.wandb_save_dir,
            "dir": self.wandb_save_dir,
        }
        if self.wandb_kwargs is not None:
            wandb_init_kwargs.update(self.wandb_kwargs)
            wandb_args.update({"init_kwargs": wandb_init_kwargs})
        self.wandb_kwargs = wandb_args
        self.wandb_logger: WandBLogger | None = None

        # setup metrics to monitor for a bunch of callbacks
        if isinstance(self.top_k, int):
            self.top_k = [self.top_k]
        # save the target top_k
        self.target_top_k = self.top_k[0]
        logger.info(
            f"Monitor top-k value is recall@{self.target_top_k}. "
            "If you provided a list of top-k values, the first one will be used."
        )
        self.metric_to_monitor = self.metric_to_monitor.format(top_k=self.target_top_k)

        # explicitly configure some callbacks that will be needed not only by the
        # Composer Trainer but also in this class
        checkpoint_kwargs = dict(
            folder=self.checkpoint_dir,
            filename=self.checkpoint_filename,
            save_interval=self.eval_interval,
            num_checkpoints_to_keep=self.save_top_k,
            monitor=self.metric_to_monitor,
            mode=self.monitor_mode,
        )
        if self.checkpoint_kwargs is not None:
            checkpoint_kwargs.update(self.checkpoint_kwargs)
        self.checkpoint_kwargs = checkpoint_kwargs
        self.model_checkpoint_callback: MetricCheckpointSaver | None = None
        self.checkpoint_path: str | os.PathLike | None = None
        # last checkpoint callback
        self.latest_model_checkpoint_callback: MetricCheckpointSaver | None = None
        self.last_checkpoint_kwargs: dict | None = None
        if self.save_last:
            last_checkpoint_kwargs = deepcopy(self.checkpoint_kwargs)
            last_checkpoint_kwargs["num_checkpoints_to_keep"] = 1
            last_checkpoint_kwargs["filename"] = (
                "last-ep{epoch}-ba{batch}-rank{rank}.pt"
            )
            self.last_checkpoint_kwargs = last_checkpoint_kwargs

        # early stopping callback
        early_stopping_kwargs = dict(
            monitor=self.metric_to_monitor,
            dataloader_label="validate",
            patience=self.early_stopping_patience,
        )
        if self.early_stopping_kwargs is not None:
            early_stopping_kwargs.update(self.early_stopping_kwargs)
        self.early_stopping_kwargs = early_stopping_kwargs
        self.early_stopping_callback: EarlyStopper | None = None

        # other callbacks declaration
        self.callbacks_store: List[composer.Callback] = []
        # add default callbacks
        self.callbacks_store += [SpeedMonitor(), LRMonitor(), MemoryMonitor()]

        # algorithms declaration
        self.algorithms: List[Algorithm] = []
        if self.gradient_clip_norm > 0:
            self.algorithms.append(
                GradientClipping(
                    clipping_type="norm", clipping_threshold=self.gradient_clip_norm
                )
            )

        # lazy trainer declaration
        self.trainer: ComposerTrainer | None = None

    @staticmethod
    def dataset_builder(
        dataset: str | GoldenRetrieverStreamingDataset = None,
        name: str = None,
        batch_size: int = None,
        question_tokenizer: PreTrainedTokenizerBase = None,
        passage_tokenizer: PreTrainedTokenizerBase = None,
        shuffle: bool = None,
        shuffle_seed: int = None,
        dataset_kwargs: dict = None,
    ):
        dataset = dataset or dataset_kwargs.get("local", None)
        if dataset is None:
            raise ValueError("The dataset is required.")
        if isinstance(dataset, str):
            # check if all the necessary parameters are provided
            if name is None and "name" not in dataset_kwargs:
                raise ValueError("The dataset name is required.")
            if batch_size is None and "batch_size" not in dataset_kwargs:
                raise ValueError("The batch size is required.")
            if (
                question_tokenizer is None
                and "question_tokenizer" not in dataset_kwargs
            ):
                raise ValueError("The question_tokenizer is required.")
            if passage_tokenizer is None and "passage_tokenizer" not in dataset_kwargs:
                raise ValueError("The passage_tokenizer is required.")
            if shuffle is None and "shuffle" not in dataset_kwargs:
                raise ValueError("The shuffle parameter is required.")
            if shuffle_seed is None and "shuffle_seed" not in dataset_kwargs:
                raise ValueError("The shuffle_seed parameter is required.")

            if "name" not in dataset_kwargs:
                dataset_kwargs["name"] = name
            if "local" not in dataset_kwargs:
                dataset_kwargs["local"] = dataset
            # if "split" not in dataset_kwargs:
            # TODO:
            # dataset_kwargs["split"] = "train"
            if "question_tokenizer" not in dataset_kwargs:
                dataset_kwargs["question_tokenizer"] = question_tokenizer
            if "passage_tokenizer" not in dataset_kwargs:
                dataset_kwargs["passage_tokenizer"] = passage_tokenizer
            if "batch_size" not in dataset_kwargs:
                dataset_kwargs["batch_size"] = batch_size
            if "shuffle" not in dataset_kwargs:
                dataset_kwargs["shuffle"] = shuffle
            if "shuffle_seed" not in dataset_kwargs:
                dataset_kwargs["shuffle_seed"] = shuffle_seed
            dataset = GoldenRetrieverStreamingDataset(**dataset_kwargs)

        return dataset, dataset_kwargs

    def configure_dataset_and_dataloader(self, *args, **kwargs):
        # dataset declaration
        self.train_dataset, self.train_dataset_kwargs = self.dataset_builder(
            dataset=self.train_dataset,
            name="train_dataset",
            batch_size=self.train_batch_size,
            question_tokenizer=self.retriever.question_tokenizer,
            passage_tokenizer=self.retriever.passage_tokenizer,
            shuffle=True,
            shuffle_seed=self.seed,
            dataset_kwargs=self.train_dataset_kwargs,
        )
        # dataloader declaration
        train_collator = GoldenRetrieverCollator(
            tokenizer=self.train_dataset.question_tokenizer
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=train_collator,
            batch_size=self.train_dataset.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=(
                max(1, 8 * self.train_dataset.batch_size // self.num_workers)
                if self.num_workers > 0
                else None
            ),
            persistent_workers=True if self.num_workers > 0 else False,
            timeout=0,
        )
        self.train_dataloader = DataSpec(
            self.train_dataloader,
            split_batch=train_collator.split_batch,
            get_num_samples_in_batch=train_collator.get_num_samples_in_batch,
            get_num_tokens_in_batch=train_collator.get_num_tokens_in_batch,
        )

        # val dataset declaration
        if self.val_dataset is not None and isinstance(
            self.val_dataset, (GoldenRetrieverStreamingDataset, str)
        ):
            self.val_dataset = [self.val_dataset]
        if isinstance(self.val_batch_size, int):
            self.val_batch_size = [self.val_batch_size] * len(self.val_dataset)
        if isinstance(self.val_dataset_kwargs, dict):
            self.val_dataset_kwargs = [self.val_dataset_kwargs] * len(self.val_dataset)

        self.val_dataloader = []
        # keep track also of the kwargs
        _val_dataset_kwargs = []
        for i, ds in enumerate(self.val_dataset):
            ds, ds_kwargs = self.dataset_builder(
                dataset=ds,
                name=f"val_dataset_{i}",
                batch_size=self.val_batch_size[i],
                question_tokenizer=self.retriever.question_tokenizer,
                passage_tokenizer=self.retriever.passage_tokenizer,
                shuffle=False,
                shuffle_seed=self.seed,
                dataset_kwargs=self.val_dataset_kwargs[i],
            )
            val_collator = GoldenRetrieverCollator(tokenizer=ds.question_tokenizer)
            val_dataloader = DataLoader(
                ds,
                collate_fn=val_collator,
                batch_size=ds.batch_size,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=(
                    max(1, 8 * self.train_dataset.batch_size // self.num_workers)
                    if self.num_workers > 0
                    else None
                ),
                persistent_workers=True if self.num_workers > 0 else False,
                timeout=0,
            )
            val_dataloader = DataSpec(
                val_dataloader,
                split_batch=val_collator.split_batch,
                get_num_samples_in_batch=val_collator.get_num_samples_in_batch,
                get_num_tokens_in_batch=val_collator.get_num_tokens_in_batch,
            )
            self.val_dataloader.append(val_dataloader)
            # keep track of the kwargs
            _val_dataset_kwargs.append(ds_kwargs)

        # update val_dataset with the new datasets
        self.val_dataset = [
            dataspec.dataloader.dataset for dataspec in self.val_dataloader
        ]
        # update val_dataset_kwargs with the new kwargs
        self.val_dataset_kwargs = _val_dataset_kwargs

        if self.test_dataset is not None and isinstance(
            self.test_dataset, GoldenRetrieverStreamingDataset
        ):
            self.test_dataset = [self.test_dataset]

        return (
            self.train_dataset,
            self.val_dataset,
            self.train_dataloader,
            self.val_dataloader,
        )

    def configure_evaluators(self):

        self.evaluators = []

        for i, val_dataloader in enumerate(self.val_dataloader):
            # try to get the label name from the dataset if has a name attribute
            if hasattr(self.val_dataset[i], "name"):
                label = self.val_dataset[i].name
            else:
                label = f"val_dataset_{i}"
            # try to get the batch size from the dataset if has a batch_size attribute
            batch_size = None
            if hasattr(self.val_dataset[i], "batch_size"):
                batch_size = self.val_dataset[i].batch_size
            evaluator = GoldenRetrieverEvaluator(
                label=label,
                dataloader=val_dataloader,
                device_eval_microbatch_size=self.device_eval_microbatch_size
                or batch_size,
            )
            self.evaluators.append(evaluator)

        return self.evaluators

    def configure_composer_module(self, *args, **kwargs):
        # # check if Index is empty
        # if len(self.retriever.document_index) == 0:
        #     # add the docs from the datasets
        #     logger.info("Document Index is empty. Adding documents from the datasets.")
        #     documents = self.retriever.document_index.documents
        #     for sample in tqdm(self.train_dataset, desc="Adding documents from train"):
        #         [documents.add_document(s) for s in sample["positives"]]
        #         [documents.add_document(s) for s in sample["negatives"]]
        #         [documents.add_document(s) for s in sample["hard_negatives"]]

        #     if self.val_dataset is not None:
        #         val_passages = []
        #         for ds in self.val_dataset:
        #             for sample in ds:
        #                 val_passages.extend(sample["positives"])
        #                 val_passages.extend(sample["negatives"])
        #                 val_passages.extend(sample["hard_negatives"])
        #         for sample in tqdm(val_passages, desc="Adding documents from val"):
        #             documents.add_document(sample)

        #     if self.test_dataset is not None:
        #         test_passages = []
        #         for ds in self.test_dataset:
        #             for sample in ds:
        #                 test_passages.extend(sample["positives"])
        #                 test_passages.extend(sample["negatives"])
        #                 test_passages.extend(sample["hard_negatives"])
        #         for sample in tqdm(test_passages, desc="Adding documents from test"):
        #             documents.add_document(sample)

        # add loss object to the retriever
        if self.retriever.loss_type is None:
            self.retriever.loss_type = self.loss()

        # lightning module declaration
        self.composer_module = GoldenRetrieverComposerModule(
            model=self.retriever, *args, **kwargs
        )

        return self.composer_module

    def configure_optimizers(self, *args, **kwargs):
        # check if it is the class or the instance
        if isinstance(self.optimizer, type):
            param_optimizer = list(self.retriever.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if "layer_norm_layer" in n
                    ],
                    "weight_decay": self.weight_decay,
                    "lr": 1e-4,
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if all(nd not in n for nd in no_decay)
                        and "layer_norm_layer" not in n
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if "layer_norm_layer" not in n
                        and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = self.optimizer(
                # params=self.retriever.parameters(),
                params=optimizer_grouped_parameters,
                lr=self.lr,
                # weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = self.optimizer

        # LR Scheduler declaration
        # check if it is the class, the instance or a function
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, type):
                self.lr_scheduler = self.lr_scheduler(**self.lr_kwargs)
            if not isinstance(
                self.lr_scheduler, composer.optim.scheduler.ComposerScheduler
            ):
                raise ValueError(
                    "The LR Scheduler should be an instance of `ComposerScheduler`"
                )

        return self.optimizer, self.lr_scheduler

    @staticmethod
    def configure_logger(
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_artifacts: bool = False,
        rank_zero_only: bool = True,
        init_kwargs: Optional[Dict[str, Any]] = None,
        online: bool = False,
    ) -> WandBLogger:
        """
        Configure the wandb logger

        Args:


        Returns:
        """
        if not online:
            os.environ["WANDB_MODE"] = "dryrun"
        wandb_logger = WandBLogger(
            project=project,
            group=group,
            name=name,
            entity=entity,
            tags=tags,
            log_artifacts=log_artifacts,
            rank_zero_only=rank_zero_only,
            init_kwargs=init_kwargs,
        )
        return wandb_logger

    @staticmethod
    def configure_early_stopping(
        monitor: str,
        dataloader_label: str,
        patience: int = 3,
        *args,
        **kwargs,
    ) -> EarlyStopper:
        logger.info(f"Enabling EarlyStopping callback with patience: {patience}")
        early_stopping_callback = EarlyStopper(
            monitor=monitor,
            dataloader_label=dataloader_label,
            patience=patience,
            *args,
            **kwargs,
        )
        return early_stopping_callback

    def configure_model_checkpoint(
        self,
        folder: Union[str, os.PathLike],
        filename: Union[str, os.PathLike],
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]],
        num_checkpoints_to_keep: int,
        monitor: Optional[str] = None,
        mode: Optional[str] = None,
        *args,
        **kwargs,
    ) -> CheckpointSaver:
        logger.info("Enabling Model Checkpointing")
        if folder is None:
            folder = (
                self.checkpoint_dir / "checkpoints" if self.checkpoint_dir else None
            )
        if filename is None:
            filename = (
                "ep{epoch}-ba{batch}-rank{rank}-" + monitor + "_{" + monitor + "}.pt"
            )
        self.checkpoint_dir = folder / filename if folder is not None else None
        logger.info(f"Checkpoint directory: {folder}")
        logger.info(f"Checkpoint filename: {filename}")

        kwargs = dict(
            folder=folder,
            filename=filename,
            save_interval=save_interval,
            num_checkpoints_to_keep=num_checkpoints_to_keep,
            monitor=monitor,
            mode=mode,
            *args,
            **kwargs,
        )
        self.model_checkpoint_callback = MetricCheckpointSaver(**kwargs)
        return self.model_checkpoint_callback

    def configure_prediction_callbacks(
        self,
        batch_size: int = 64,
        precision: int | str = 32,
        k: int | None = None,
        force_reindex: bool = True,
        interval: Union[int, str, Time, Callable[[State, Event], bool]] = "1ep",
        metrics_callbacks: list[NLPTemplateCallback] | None = None,
        *args,
        **kwargs,
    ):
        if k is None:
            # we need the largest k for the prediction callback
            # get the max top_k for the prediction callback
            k = sorted(self.top_k, reverse=True)[0]
        if metrics_callbacks is None:
            # TODO: convert the metrics_callbacks to the new callbacks
            metrics_callbacks = self.configure_metrics_callbacks()

        prediction_callback = PredictionCallback(
            k=k,
            batch_size=batch_size,
            precision=precision,
            force_reindex=force_reindex,
            metric_callbacks=metrics_callbacks,
            interval=interval,
        )

        return prediction_callback

    def configure_metrics_callbacks(
        self, save_predictions: bool = False
    ) -> List[NLPTemplateCallback]:
        """
        Configure the metrics callbacks for the trainer. This method is called
        by the `eval_callbacks` method, and it is used to configure the callbacks
        that will be used to evaluate the model during training.

        Args:
            save_predictions (`bool`, optional, defaults to `False`):
                Whether to save the predictions to disk or not

        Returns:
            `List[NLPTemplateCallback]`:
                The list of callbacks to use for evaluation
        """
        # prediction callback
        # metrics_callbacks: List[NLPTemplateCallback] = [
        #     RecallAtKEvaluationCallback(k, verbose=True) for k in self.top_k
        # ]
        # metrics_callbacks += [
        #     AvgRankingEvaluationCallback(k, verbose=True) for k in self.top_k
        # ]
        # if save_predictions:
        #     metrics_callbacks.append(SavePredictionsCallback())
        metrics_callbacks: List[NLPTemplateCallback] = [
            ComposerRecallAtKEvaluationCallback(k, verbose=True) for k in self.top_k
        ]
        return metrics_callbacks

    def configure_hard_negatives_callback(self):
        metrics_to_monitor = (
            self.metrics_to_monitor_for_hard_negatives or self.metric_to_monitor
        )
        hn_dataset, _ = self.dataset_builder(dataset_kwargs=self.train_dataset_kwargs)
        dataloader = DataLoader(
            hn_dataset,
            collate_fn=GoldenRetrieverCollator(tokenizer=hn_dataset.question_tokenizer),
            batch_size=hn_dataset.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            # pin_memory=True,
            # prefetch_factor=2,
        )
        hard_negatives_callback = HardNegativeMiningCallback(
            k=self.target_top_k,
            batch_size=self.prediction_batch_size,
            precision=self.precision,
            interval=self.eval_interval,
            dataloader=dataloader,
            metrics_to_monitor=metrics_to_monitor,
            threshold=self.hard_negatives_threshold,
            max_negatives=self.max_hard_negatives_to_mine,
            add_with_probability=self.mine_hard_negatives_with_probability,
        )
        hard_negative_algo = HardNegativeAlgorithm(
            self.train_dataset.passage_tokenizer,
            max_length=self.train_dataset.max_passage_length,
        )
        self.algorithms.append(hard_negative_algo)

        return hard_negatives_callback

    def training_callbacks(self):
        if self.model_checkpointing:
            self.model_checkpoint_callback = self.configure_model_checkpoint(
                **self.checkpoint_kwargs
            )
            self.callbacks_store.append(self.model_checkpoint_callback)
            if self.save_last:
                self.latest_model_checkpoint_callback = self.configure_model_checkpoint(
                    **self.last_checkpoint_kwargs
                )
                self.callbacks_store.append(self.latest_model_checkpoint_callback)

            # self.callbacks_store.append(SaveRetrieverCallback())
        if self.early_stopping:
            self.early_stopping_callback = self.configure_early_stopping(
                **self.early_stopping_kwargs
            )
        return self.callbacks_store

    def train(self, *args, **kwargs):
        """
        Train the model

        Args:
            *args:
                Additional args
            **kwargs:
                Additional kwargs

        Returns:
            `None`
        """
        if self.log_to_wandb:
            logger.info("Instantiating Wandb Logger")
            # log the args to wandb
            self.wandb_logger = self.configure_logger(**self.wandb_kwargs)

        # add the evaluation callbacks
        self.callbacks_store.append(
            self.configure_prediction_callbacks(
                batch_size=self.prediction_batch_size, precision=self.precision
            )
        )
        # add the hard negatives callback after the evaluation callback
        if self.max_hard_negatives_to_mine > 0:
            self.callbacks_store.append(self.configure_hard_negatives_callback())

        # set-up training specific callbacks
        self.callbacks_store = self.training_callbacks()

        # self.callbacks_store.append(FreeUpIndexerVRAMCallback())

        if self.trainer is None:
            logger.info("Instantiating the Trainer")
            self.trainer = ComposerTrainer(
                model=self.composer_module,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.evaluators,
                device_train_microbatch_size=self.device_train_microbatch_size
                or self.train_dataset.batch_size,
                algorithms=self.algorithms,
                progress_bar=self.progress_bar,
                log_to_console=self.log_to_console,
                device=self.device,
                precision=COMPOSER_PRECISION_INPUT_STR_ALIAS_CONVERSION.get(
                    self.precision, self.precision
                ),
                optimizers=self.optimizer,
                schedulers=self.lr_scheduler,
                step_schedulers_every_batch=self.step_schedulers_every_batch,  # interval should be step
                max_duration=self.max_duration,
                eval_interval=self.eval_interval,
                dist_timeout=self.dist_timeout,
                load_path=self.resume_from_checkpoint_path,
                seed=self.seed,
                callbacks=self.callbacks_store,
                loggers=self.wandb_logger,
                deepspeed_config=self.deepspeed_config,
                fsdp_config=self.fsdp_config,
                # deepspeed_config={
                #     "train_batch_size": 64,
                #     "train_micro_batch_size_per_gpu": 32,
                #     "gradient_accumulation_steps": 1,
                #     "bf16": {"enabled": True},
                #     "zero_optimization": {
                #         "stage": 1,
                #         # "offload_optimizer": {"device": "cpu", "pin_memory": True},
                #     },
                # },
                **self.composer_trainer_kwargs,
            )

        self.trainer.fit()

    def test(
        self,
        composer_module: ComposerModel | None = None,
        checkpoint_path: str | os.PathLike | None = None,
        test_dataloader: DataLoader | Evaluator | None = None,
        force_reindex: bool = False,
        *args,
        **kwargs,
    ):
        """
        Test the model

        Args:
            lightning_module (`GoldenRetrieverPLModule`, optional, defaults to `None`):
                The lightning module to test
            checkpoint_path (`str`, `os.PathLike`, optional, defaults to `None`):
                The path to the checkpoint to load
            lightning_datamodule (`GoldenRetrieverPLDataModule`, optional, defaults to `None`):
                The lightning data module to use for testing
            *args:
                Additional args
            **kwargs:
                Additional kwargs

        Returns:
            `None`
        """
        raise NotImplementedError("The `test` method is not implemented yet.")
        # if self.test_dataset is None:
        #     logger.warning("No test dataset provided. Skipping testing.")
        #     return

        # if self.trainer is None:
        #     self.trainer = ComposerTrainer(
        #         model=self.composer_module,
        #         train_dataloader=self.train_dataloader,
        #         eval_dataloader=self.evaluators,
        #         device_train_microbatch_size=self.device_train_microbatch_size
        #         or self.train_dataset.batch_size,
        #         algorithms=self.algorithms,
        #         progress_bar=self.progress_bar,
        #         log_to_console=self.log_to_console,
        #         device=self.device,
        #         precision=COMPOSER_PRECISION_INPUT_STR_ALIAS_CONVERSION.get(
        #             self.precision, self.precision
        #         ),
        #         optimizers=self.optimizer,
        #         schedulers=self.lr_scheduler,
        #         step_schedulers_every_batch=self.step_schedulers_every_batch,  # interval should be step
        #         max_duration=self.max_duration,
        #         eval_interval=self.eval_interval,
        #         dist_timeout=self.dist_timeout,
        #         load_path=self.resume_from_checkpoint_path,
        #         seed=self.seed,
        #         callbacks=self.callbacks_store,
        #         loggers=self.wandb_logger,
        #         deepspeed_config=self.deepspeed_config,
        #         fsdp_config=self.fsdp_config,
        #         # deepspeed_config={
        #         #     "train_batch_size": 64,
        #         #     "train_micro_batch_size_per_gpu": 32,
        #         #     "gradient_accumulation_steps": 1,
        #         #     "bf16": {"enabled": True},
        #         #     "zero_optimization": {
        #         #         "stage": 1,
        #         #         # "offload_optimizer": {"device": "cpu", "pin_memory": True},
        #         #     },
        #         # },
        #         **self.composer_trainer_kwargs,
        #     # self.trainer = pl.Trainer(
        #     #     accelerator=self.accelerator,
        #     #     devices=self.devices,
        #     #     num_nodes=self.num_nodes,
        #     #     strategy=self.strategy,
        #     #     deterministic=self.deterministic,
        #     #     fast_dev_run=self.fast_dev_run,
        #     #     precision=self.precision,
        #     #     callbacks=[
        #     #         self.configure_prediction_callbacks(
        #     #             batch_size=self.prediction_batch_size,
        #     #             precision=self.precision,
        #     #             force_reindex=force_reindex,
        #     #         )
        #     #     ],
        #     #     **self.composer_trainer_kwargs,
        #     )
        # if lightning_module is not None:
        #     best_lightning_module = lightning_module
        # else:
        #     try:
        #         if self.fast_dev_run:
        #             best_lightning_module = self.composer_module
        #         else:
        #             # load best model for testing
        #             if checkpoint_path is not None:
        #                 best_model_path = checkpoint_path
        #             elif self.checkpoint_path is not None:
        #                 best_model_path = self.checkpoint_path
        #             elif self.model_checkpoint_callback:
        #                 best_model_path = self.model_checkpoint_callback.best_model_path
        #             else:
        #                 raise ValueError(
        #                     "Either `checkpoint_path` or `model_checkpoint_callback` should "
        #                     "be provided to the trainer"
        #                 )
        #             logger.info(f"Loading best model from {best_model_path}")

        #             best_lightning_module = (
        #                 GoldenRetrieverPLModule.load_from_checkpoint(best_model_path)
        #             )
        #     except Exception as e:
        #         logger.info(f"Failed to load the model from checkpoint: {e}")
        #         logger.info("Using last model instead")
        #         best_lightning_module = self.composer_module

        # lightning_datamodule = lightning_datamodule or self.lightning_datamodule
        # # module test
        # self.trainer.test(best_lightning_module, datamodule=lightning_datamodule)

    def convert_to_yaml(self):
        return OmegaConf.to_yaml(cfg=to_config(self))

    @classmethod
    def to_config(cls):
        config = {
            "_target_": f"{cls.__class__.__module__}.{cls.__class__.__name__}",
            "retriever": to_config(cls.retriever),
            "train_dataset": (
                to_config(cls.train_dataset) if cls.train_dataset is not None else None
            ),
            "val_dataset": (
                to_config(cls.val_dataset) if cls.val_dataset is not None else None
            ),
            "test_dataset": (
                to_config(cls.test_dataset) if cls.test_dataset is not None else None
            ),
            "num_workers": cls.num_workers,
            # trainer parameters
            "optimizer": to_config(cls.optimizer),
            "lr": cls.lr,
            "weight_decay": cls.weight_decay,
            "lr_scheduler": to_config(cls.lr_scheduler),
            "num_warmup_steps": cls.num_warmup_steps,
            "loss": to_config(cls.loss),
            "callbacks": (
                to_config(cls.callbacks) if cls.callbacks is not None else None
            ),
            "accelerator": cls.accelerator,
            "devices": cls.devices,
            "num_nodes": cls.num_nodes,
            "strategy": cls.strategy,
            "accumulate_grad_batches": cls.accumulate_grad_batches,
            "gradient_clip_val": cls.gradient_clip_val,
            "val_check_interval": cls.val_check_interval,
            "check_val_every_n_epoch": cls.check_val_every_n_epoch,
            "max_steps": cls.max_steps,
            "max_epochs": cls.max_epochs,
            "deterministic": cls.deterministic,
            "fast_dev_run": cls.fast_dev_run,
            "precision": cls.precision,
            "reload_dataloaders_every_n_epochs": cls.reload_dataloaders_every_n_epochs,
            "trainer_kwargs": to_config(cls.composer_trainer_kwargs),
        }


@hydra.main(config_path="../../conf", config_name="default", version_base="1.3")
def main(conf: omegaconf.DictConfig):
    train_hydra(conf)


def train_hydra(conf: omegaconf.DictConfig) -> None:
    # TODO: add the ability to pass the config file
    pass


if __name__ == "__main__":
    main()