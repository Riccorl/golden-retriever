import datetime
import os
import pprint
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal

import hydra
import lightning as L
import omegaconf
import torch
import transformers as tr
from composer.utils import dist, get_device, reproducibility
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops

# from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    # LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from pprintpp import pformat
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from tqdm import tqdm

from goldenretriever.callbacks.base import NLPTemplateCallback
from goldenretriever.callbacks.evaluation_callbacks import (
    AvgRankingEvaluationCallback,
    RecallAtKEvaluationCallback,
)
from goldenretriever.callbacks.prediction_callbacks import (
    GoldenRetrieverFabricPredictionCallback,
    GoldenRetrieverPredictionCallback,
)
from goldenretriever.callbacks.training_callbacks import NegativeAugmentationCallback, NegativeAugmentationFabricCallback
from goldenretriever.callbacks.utils_callbacks import (
    FreeUpIndexerVRAMCallback,
    LearningRateMonitor,
    SavePredictionsCallback,
    SaveRetrieverCallback,
)
from goldenretriever.common.from_config import FromConfig
from goldenretriever.common.log import get_logger
from goldenretriever.common.utils import get_callable_from_string, to_config
from goldenretriever.data.datasets import GoldenRetrieverDataset
from goldenretriever.data.streaming_dataset import GoldenRetrieverStreamingDataset
from goldenretriever.indexers.base import BaseDocumentIndex
from goldenretriever.lightning_modules.pl_data_modules import (
    GoldenRetrieverPLDataModule,
)
from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule
from goldenretriever.pytorch_modules.loss import MultiLabelNCELoss
from goldenretriever.pytorch_modules.model import GoldenRetriever
from goldenretriever.pytorch_modules.optim import RAdamW
from goldenretriever.pytorch_modules.scheduler import LinearScheduler
from goldenretriever.trainer.custom_trainer import MyCustomTrainer
from goldenretriever.trainer.utils import (
    PRECISION_INPUT_STR_ALIAS_CONVERSION,
    CycleIterator,
)

logger = get_logger(__name__)

import inspect
from dataclasses import asdict, is_dataclass

# from torch import distributed as dist
from composer.utils import dist
from lightning.pytorch.utilities.types import LRSchedulerConfig
from streaming.base.distributed import get_rank, get_world_size
from torch.optim.lr_scheduler import LRScheduler


def maybe_init_dist(timeout: float = 300.0) -> bool:
    """Initialize torch.distributed ourselves, if necessary.

    Returns:
        bool: Whether we initialized dist ourselves.
    """
    timeout_timedelta = datetime.timedelta(seconds=timeout)
    if get_world_size() == 1 or not dist.is_available() or dist.is_initialized():
        return False
    if torch.cuda.is_available() and dist.is_nccl_available():
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(
        backend=backend,
        rank=get_rank(),
        world_size=get_world_size(),
        timeout=timeout_timedelta,
    )
    return True


def capture_hparams() -> Dict[str, Any]:
    """Captures the local variables ('hyperparameters') from where this function gets called."""
    caller_frame = inspect.currentframe().f_back
    locals_of_caller = caller_frame.f_locals
    hparams = {}
    for name, value in locals_of_caller.items():
        if value is None or isinstance(value, (int, float, str, bool, Path)):
            hparams[name] = value
        elif is_dataclass(value):
            hparams[name] = asdict(value)
        else:
            hparams[name] = str(value)
    return hparams


class FabricTrainer(FromConfig):
    def __init__(
        self,
        retriever: GoldenRetriever | DictConfig,
        document_index: (
            str
            | BaseDocumentIndex
            | DictConfig
            | List[str]
            | List[BaseDocumentIndex]
            | List[DictConfig]
            | None
        ) = None,
        train_dataset: str | GoldenRetrieverStreamingDataset | None = None,
        global_batch_size: int = 32,
        train_micro_batch_size: int = 32,
        train_dataset_kwargs: dict | None = None,
        val_dataset: (
            str
            | List[str]
            | GoldenRetrieverStreamingDataset
            | list[GoldenRetrieverStreamingDataset]
            | None
        ) = None,
        val_batch_size: int = 32,
        val_dataset_kwargs: dict | list[dict] | None = None,
        test_dataset: (
            str
            | List[str]
            | GoldenRetrieverDataset
            | list[GoldenRetrieverDataset]
            | None
        ) = None,
        test_batch_size: int = 32,
        test_dataset_kwargs: dict | list[dict] | None = None,
        num_workers: int = 4,
        add_documents_from_dataset: bool = False,
        optimizer: torch.optim.Optimizer = RAdamW,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = LinearScheduler,
        num_warmup_steps: int = 0,
        micro_batch_size: int | None = None,
        callbacks: list | None = None,
        accelerator: str = "auto",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "auto",
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 1.0,
        val_check_interval: float = 1.0,
        check_val_every_n_epoch: int = 1,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        deterministic: bool = True,
        fast_dev_run: bool = False,
        precision: int | str = 16,
        compile: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        resume_from_checkpoint_path: str | os.PathLike | None = None,
        lightning_trainer_kwargs: dict | None = None,
        # eval parameters
        metric_to_monitor: str = "validate_recall@{top_k}",
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
        wandb_log_model: bool = False,
        wandb_online_mode: bool = False,
        wandb_watch: str = "all",
        wandb_kwargs: dict | None = None,
        # checkpoint parameters
        model_checkpointing: bool = True,
        checkpoint_dir: str | os.PathLike = "./checkpoints",
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
        seed: int = 42,
        float32_matmul_precision: str = "medium",
        **kwargs,
    ):
        dist.initialize_dist(get_device(None), timeout=300.0)
        # put all the parameters in the class
        self.retriever = retriever
        self.document_index = document_index
        # datasets
        self.train_dataset = train_dataset
        self.global_batch_size = global_batch_size
        self.train_micro_batch_size = train_micro_batch_size
        self.train_dataset_kwargs = train_dataset_kwargs or {}
        self.val_dataset = val_dataset
        self.val_batch_size = val_batch_size
        self.val_dataset_kwargs = val_dataset_kwargs or {}
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size
        self.test_dataset_kwargs = test_dataset_kwargs or {}
        self.num_workers = num_workers
        self.add_documents_from_dataset = add_documents_from_dataset
        # trainer parameters
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.micro_batch_size = micro_batch_size
        self.callbacks = callbacks
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.deterministic = deterministic
        self.fast_dev_run = fast_dev_run
        self.precision = precision
        self.compile = compile
        self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.resume_from_checkpoint_path = resume_from_checkpoint_path
        self.trainer_kwargs = lightning_trainer_kwargs or {}
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
        self.wandb_log_model = wandb_log_model
        self.wandb_online_mode = wandb_online_mode
        self.wandb_watch = wandb_watch
        self.wandb_kwargs = wandb_kwargs
        # checkpoint parameters
        self.model_checkpointing = model_checkpointing
        self.checkpoint_dir = checkpoint_dir
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
        self.seed = seed
        self.float32_matmul_precision = float32_matmul_precision

        # reproducibility
        L.seed_everything(self.seed)
        # set the precision of matmul operations
        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        # TODO: check if can be fixed and saved automatically to a file
        self.hparams = capture_hparams()

        # check strategy if it is a string or a class or a DictConfig
        if isinstance(self.strategy, DictConfig):
            self.strategy = hydra.utils.instantiate(self.strategy)

        # print(hparams)

        if self.max_epochs is None and self.max_steps is None:
            raise ValueError(
                "Either `max_epochs` or `max_steps` should be specified in the trainer configuration"
            )

        if self.max_epochs is not None and self.max_steps is not None:
            logger.info(
                "Both `max_epochs` and `max_steps` are specified in the trainer configuration. "
                "Will use `max_epochs` for the number of training steps"
            )
            self.max_steps = None

        # check if the retriever is a DictConfig
        # self.retriever = self.configure_retriever()

        # lightning data module declaration
        # self.lightning_datamodule = self.configure_lightning_datamodule()
        self.lightning_datamodule: GoldenRetrieverPLDataModule | None = None

        # if self.max_epochs is not None:
        #     logger.info(f"Number of training epochs: {self.max_epochs}")
        #     self.max_steps = (
        #         len(self.lightning_datamodule.train_dataloader()) * self.max_epochs
        #     )

        # if self.add_documents_from_dataset:
        #     self.add_documents_from_dataset()

        # optimizer declaration
        # self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # lightning module declaration
        # self.lightning_module = self.configure_lightning_module()

        # logger and experiment declaration
        # update self.wandb_kwargs
        wandb_args = dict(
            entity=self.wandb_entity,
            project=self.wandb_project_name,
            name=self.wandb_experiment_name,
            save_dir=self.wandb_save_dir,
            log_model=self.wandb_log_model,
            offline=not self.wandb_online_mode,
            watch=self.wandb_watch,
            # lightning_module=self.lightning_module,
        )
        if self.wandb_kwargs is not None:
            wandb_args.update(self.wandb_kwargs)
        self.wandb_kwargs = wandb_args
        self.wandb_logger: WandbLogger | None = None
        self.experiment_path: Path | None = None

        # setup metrics to monitor for a bunch of callbacks
        if isinstance(self.top_k, int):
            self.top_k = [self.top_k]
        # save the target top_k
        self.target_top_k = self.top_k[0]
        self.metric_to_monitor = self.metric_to_monitor.format(top_k=self.target_top_k)

        # explicitly configure some callbacks that will be needed not only by the
        # L.Trainer but also in this class
        # model checkpoint callback
        if self.save_last:
            logger.warning(
                "We will override the `save_last` of `ModelCheckpoint` to `False`. "
                "Instead, we will use a separate `ModelCheckpoint` callback to save the last checkpoint"
            )
        checkpoint_kwargs = dict(
            monitor=self.metric_to_monitor,
            mode=self.monitor_mode,
            verbose=True,
            save_top_k=self.save_top_k,
            filename=self.checkpoint_filename,
            dirpath=self.checkpoint_dir,
            auto_insert_metric_name=False,
        )
        if self.checkpoint_kwargs is not None:
            checkpoint_kwargs.update(self.checkpoint_kwargs)
        self.checkpoint_kwargs = checkpoint_kwargs
        self.model_checkpoint_callback: ModelCheckpoint | None = None
        self.checkpoint_path: str | os.PathLike | None = None
        # last checkpoint callback
        self.latest_model_checkpoint_callback: ModelCheckpoint | None = None
        self.last_checkpoint_kwargs: dict | None = None
        if self.save_last:
            last_checkpoint_kwargs = deepcopy(self.checkpoint_kwargs)
            last_checkpoint_kwargs["save_top_k"] = 1
            last_checkpoint_kwargs["filename"] = "last-{epoch}-{step}"
            last_checkpoint_kwargs["monitor"] = "step"
            last_checkpoint_kwargs["mode"] = "max"
            self.last_checkpoint_kwargs = last_checkpoint_kwargs

        # early stopping callback
        early_stopping_kwargs = dict(
            monitor=self.metric_to_monitor,
            mode=self.monitor_mode,
            patience=self.early_stopping_patience,
        )
        if self.early_stopping_kwargs is not None:
            early_stopping_kwargs.update(self.early_stopping_kwargs)
        self.early_stopping_kwargs = early_stopping_kwargs
        self.early_stopping_callback: EarlyStopping | None = None

        # other callbacks declaration
        self.callbacks_store: List[L.Callback] = []  # self.configure_callbacks()
        # add default callbacks
        self.callbacks_store += [
            # ModelSummary(max_depth=2),
            # LearningRateMonitor(logging_interval="step"),
        ]

        # lazy trainer declaration
        self.trainer: L.Trainer | None = None
        self.fabric: L.Fabric | None = None
        # it is used to store the state of the trainer
        self.state = {}

        # if devices > 1:
        #     strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
        # else:
        #     strategy = "auto"

    def configure_lightning_datamodule(self, *args, **kwargs):

        self.train_dataset_kwargs["batch_size"] = self.train_micro_batch_size

        # lightning data module declaration
        if self.val_dataset is not None and isinstance(
            self.val_dataset,
            (GoldenRetrieverDataset, GoldenRetrieverStreamingDataset, str),
        ):
            self.val_dataset = [self.val_dataset]
            self.val_dataset_kwargs["batch_size"] = self.val_batch_size
            self.val_batch_size = [self.val_batch_size]
            self.val_dataset_kwargs = [self.val_dataset_kwargs]
        if self.test_dataset is not None and isinstance(
            self.test_dataset,
            (GoldenRetrieverDataset, GoldenRetrieverStreamingDataset, str),
        ):
            self.test_dataset = [self.test_dataset]
            self.test_dataset_kwargs["batch_size"] = self.test_batch_size
            self.test_batch_size = [self.test_batch_size]
            self.test_dataset_kwargs = [self.test_dataset_kwargs]

        self.lightning_datamodule = GoldenRetrieverPLDataModule(
            train_dataset=self.train_dataset,
            train_dataset_kwargs=self.train_dataset_kwargs,
            val_datasets=self.val_dataset,
            val_datasets_kwargs=self.val_dataset_kwargs,
            test_datasets=self.test_dataset,
            test_datasets_kwargs=self.test_dataset_kwargs,
            num_workers=self.num_workers,
            question_tokenizer=self.retriever.question_tokenizer,
            passage_tokenizer=self.retriever.passage_tokenizer,
            *args,
            **kwargs,
        )
        # with self.fabric.rank_zero_first():
        self.lightning_datamodule.prepare_data()
        self.lightning_datamodule.setup()
        return self.lightning_datamodule

    def configure_retriever(self, *args, **kwargs):
        # check if the retriever is a DictConfig
        if isinstance(self.retriever, omegaconf.DictConfig):
            self.retriever = hydra.utils.instantiate(self.retriever, *args, **kwargs)

        return self.retriever

    def add_documents_from_dataset(self, add_test: bool = False):
        # # check if Index is empty
        # if len(self.retriever.document_index) == 0 or force:
        logger.info("Adding documents from the datasets to the Index")
        # add the docs from the datasets
        logger.info("Document Index is empty. Adding documents from the datasets.")
        documents = self.retriever.document_index.documents
        for sample in tqdm(self.train_dataset, desc="Adding documents from train"):
            [documents.add_document(s) for s in sample["positives"]]
            [documents.add_document(s) for s in sample["negatives"]]
            [documents.add_document(s) for s in sample["hard_negatives"]]

        if self.val_dataset is not None:
            val_passages = []
            for ds in self.val_dataset:
                for sample in ds:
                    val_passages.extend(sample["positives"])
                    val_passages.extend(sample["negatives"])
                    val_passages.extend(sample["hard_negatives"])
            for sample in tqdm(val_passages, desc="Adding documents from val"):
                documents.add_document(sample)

        if self.test_dataset is not None and add_test:
            test_passages = []
            for ds in self.test_dataset:
                for sample in ds:
                    test_passages.extend(sample["positives"])
                    test_passages.extend(sample["negatives"])
                    test_passages.extend(sample["hard_negatives"])
            for sample in tqdm(test_passages, desc="Adding documents from test"):
                documents.add_document(sample)

    def configure_lightning_module(self, *args, **kwargs):
        # lightning module declaration
        self.lightning_module = GoldenRetrieverPLModule(
            model=self.retriever,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            micro_batch_size=self.micro_batch_size,
            *args,
            **kwargs,
        )
        return self.lightning_module

    def configure_optimizers(self, *args, **kwargs):

        # cast the optimizer to the class
        if isinstance(self.optimizer, str):
            # convert string to a callable
            self.optimizer = get_callable_from_string(self.optimizer)

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
            if isinstance(self.lr_scheduler, str):
                # convert string to a callable
                self.lr_scheduler = get_callable_from_string(self.lr_scheduler)
            if isinstance(self.lr_scheduler, type):
                self.lr_scheduler = self.lr_scheduler(
                    optimizer=self.optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.max_steps,
                )

        return self.optimizer, self.lr_scheduler

    @staticmethod
    def configure_logger(
        name: str,
        save_dir: str | os.PathLike,
        offline: bool,
        entity: str,
        project: str,
        log_model: Literal["all"] | bool,
        watch: str | None = None,
        lightning_module: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ) -> WandbLogger:
        """
        Configure the wandb logger

        Args:
            name (`str`):
                The name of the experiment
            save_dir (`str`, `os.PathLike`):
                The directory where to save the experiment
            offline (`bool`):
                Whether to run wandb offline
            entity (`str`):
                The wandb entity
            project (`str`):
                The wandb project name
            log_model (`Literal["all"]`, `bool`):
                Whether to log the model to wandb
            watch (`str`, optional, defaults to `None`):
                The mode to watch the model
            lightning_module (`torch.nn.Module`, optional, defaults to `None`):
                The lightning module to watch
            *args:
                Additional args
            **kwargs:
                Additional kwargs

        Returns:
            `lightning.loggers.WandbLogger`:
                The wandb logger
        """
        wandb_logger = WandbLogger(
            name=name,
            save_dir=save_dir,
            offline=offline,
            project=project,
            log_model=log_model and not offline,
            entity=entity,
            *args,
            **kwargs,
        )
        if watch is not None and lightning_module is not None:
            watch_kwargs = dict(model=lightning_module)
            if watch is not None:
                watch_kwargs["log"] = watch
            wandb_logger.watch(**watch_kwargs)
        return wandb_logger

    @staticmethod
    def configure_early_stopping(
        monitor: str,
        mode: str,
        patience: int = 3,
        *args,
        **kwargs,
    ) -> EarlyStopping:
        logger.info(f"Enabling EarlyStopping callback with patience: {patience}")
        early_stopping_callback = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            *args,
            **kwargs,
        )
        return early_stopping_callback

    def configure_model_checkpoint(
        self,
        monitor: str,
        mode: str,
        verbose: bool = True,
        save_top_k: int = 1,
        save_last: bool = False,
        filename: str | os.PathLike | None = None,
        dirpath: str | os.PathLike | None = None,
        auto_insert_metric_name: bool = False,
        *args,
        **kwargs,
    ) -> ModelCheckpoint:
        logger.info("Enabling Model Checkpointing")
        if dirpath is None:
            dirpath = (
                self.experiment_path / "checkpoints" if self.experiment_path else None
            )
        else:
            dirpath = Path(dirpath) / self.wandb_project_name
        if filename is None:
            filename = (
                "checkpoint-" + monitor + "_{" + monitor + ":.4f}-epoch_{epoch:02d}"
            )
        self.checkpoint_path = dirpath / filename if dirpath is not None else None
        logger.info(f"Checkpoint directory: {dirpath}")
        logger.info(f"Checkpoint filename: {filename}")

        kwargs = dict(
            monitor=monitor,
            mode=mode,
            verbose=verbose,
            save_top_k=save_top_k,
            save_last=save_last,
            filename=filename,
            dirpath=dirpath,
            auto_insert_metric_name=auto_insert_metric_name,
            *args,
            **kwargs,
        )

        self.model_checkpoint_callback = ModelCheckpoint(**kwargs)
        return self.model_checkpoint_callback

    def configure_hard_negatives_callback(self):
        metrics_to_monitor = (
            self.metrics_to_monitor_for_hard_negatives or self.metric_to_monitor
        )
        # hard_negatives_callback = NegativeAugmentationCallback(
        hard_negatives_callback = NegativeAugmentationFabricCallback(
            k=self.target_top_k,
            batch_size=self.prediction_batch_size,
            num_workers=self.num_workers,
            precision=self.precision,
            stages=["validate"],
            metrics_to_monitor=metrics_to_monitor,
            threshold=self.hard_negatives_threshold,
            max_negatives=self.max_hard_negatives_to_mine,
            add_with_probability=self.mine_hard_negatives_with_probability,
            refresh_every_n_epochs=1,
        )
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

            self.callbacks_store.append(SaveRetrieverCallback())
        if self.early_stopping:
            self.early_stopping_callback = self.configure_early_stopping(
                **self.early_stopping_kwargs
            )
        return self.callbacks_store

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
        metrics_callbacks: List[NLPTemplateCallback] = [
            RecallAtKEvaluationCallback(k, verbose=True) for k in self.top_k
        ]
        # metrics_callbacks += [
        #     AvgRankingEvaluationCallback(k, verbose=True) for k in self.top_k
        # ]
        if save_predictions:
            metrics_callbacks.append(SavePredictionsCallback())
        return metrics_callbacks

    def configure_prediction_callbacks(
        self,
        batch_size: int = 64,
        precision: int | str = 32,
        k: int | None = None,
        force_reindex: bool = True,
        metrics_callbacks: list[NLPTemplateCallback] | None = None,
        *args,
        **kwargs,
    ):
        if k is None:
            # we need the largest k for the prediction callback
            # get the max top_k for the prediction callback
            k = sorted(self.top_k, reverse=True)[0]
        if metrics_callbacks is None:
            metrics_callbacks = self.configure_metrics_callbacks()

        # prediction_callback = GoldenRetrieverPredictionCallback(
        prediction_callback = GoldenRetrieverFabricPredictionCallback(
            batch_size=batch_size,
            precision=precision,
            k=k,
            force_reindex=force_reindex,
            other_callbacks=metrics_callbacks,
            *args,
            **kwargs,
        )
        return prediction_callback

    def init_experiment(self):
        # setup logger
        if self.log_to_wandb:
            # TODO: add CSV logger if wandb is not available
            logger.info("Instantiating Wandb Logger")
            # log the args to wandb
            # logger.info(pformat(self.wandb_kwargs))
            self.wandb_logger = self.configure_logger(**self.wandb_kwargs)
            self.experiment_path = None
            try:
                self.experiment_path = Path(self.wandb_logger.experiment.dir)
            except Exception as e:
                logger.info(f"Failed to get the experiment path: {e}")

        # set-up training specific callbacks
        # self.callbacks_store = self.training_callbacks()
        # add the evaluation callbacks
        self.callbacks_store.append(
            self.configure_prediction_callbacks(
                batch_size=self.prediction_batch_size,
                precision=self.precision,
            )
        )
        # add the hard negatives callback after the evaluation callback
        if self.max_hard_negatives_to_mine > 0:
            self.callbacks_store.append(self.configure_hard_negatives_callback())

        # self.callbacks_store.append(FreeUpIndexerVRAMCallback())

        # if self.fabric is None:
        #     self.fabric = L.Fabric(
        #         devices=self.devices,
        #         strategy=self.strategy,
        #         precision=PRECISION_INPUT_STR_ALIAS_CONVERSION.get(
        #             self.precision, self.precision
        #         ),
        #         loggers=[self.wandb_logger],
        #         callbacks=self.callbacks_store,
        #     )
        #     self.fabric.launch()

        #     self.fabric.print(pprint.pformat(self.hparams))
        #     if isinstance(self.wandb_logger, WandbLogger):
        #         self.fabric.logger.log_hyperparams(self.hparams)

        #     # same seed for every process to init model (FSDP)
        #     self.fabric.seed_everything(self.seed)

        # retriever declaration
        if isinstance(self.retriever, DictConfig):
            #     with self.fabric.init_module(empty_init=False):
            self.retriever = self.configure_retriever()
        # if self.compile:
        #     self.retriever = torch.compile(self.retriever)
        # self.retriever = self.fabric.setup(self.retriever)

        # dataloader declaration
        self.lightning_datamodule = self.configure_lightning_datamodule()
        self.train_dataloader, self.val_dataloader = (
            self.lightning_datamodule.train_dataloader(),
            self.lightning_datamodule.val_dataloader(),
        )
        if self.max_epochs is not None:
            logger.info(f"Number of training epochs: {self.max_epochs}")
            self.max_steps = (
                len(self.lightning_datamodule.train_dataloader()) * self.max_epochs
            )
        # self.train_dataloader = self.fabric.setup_dataloaders(self.train_dataloader)
        # self.val_dataloader = self.fabric.setup_dataloaders(*self.val_dataloader)

        # optimizer declaration
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        # self.optimizer = self.fabric.setup_optimizers(self.optimizer)

        if isinstance(self.lr_scheduler, LRScheduler):
            # create a LRSchedulerConfig from the LRScheduler
            self.lr_scheduler = LRSchedulerConfig(
                scheduler=self.lr_scheduler, interval="epoch", frequency=1
            )

        self.callbacks_store.append(
            LearningRateMonitor(
                optimizers=[self.optimizer],
                lr_scheduler_configs=[self.lr_scheduler],
                logging_interval="step",
            )
        )

        # add reload from checkpoint logic
        # state declaration
        # self.state = {
        #     "model": self.retriever,
        #     "optimizer": self.optimizer,
        #     "scheduler": self.lr_scheduler,
        #     "train_dataloader": self.train_dataloader,
        #     "iter_num": 0,
        #     "step_count": 0,
        #     "epoch_count": 0,
        #     "max_steps": self.max_steps,
        # }

    # def compute_gradient_accumulation_iters(self) -> int:
    #     devices = 1
    #     return self.global_batch_size // devices // self.train_micro_batch_size

    def train(self):

        self.init_experiment()

        if self.trainer is None:
            self.trainer = MyCustomTrainer(
                accelerator=self.accelerator,
                strategy=self.strategy,
                devices=self.devices,
                num_nodes=self.num_nodes,
                precision=self.precision,
                # plugins=
                callbacks=self.callbacks_store,
                loggers=[self.wandb_logger],
                max_epochs=self.max_epochs,
                max_steps=self.max_steps,
                validation_frequency=self.val_check_interval,
                seed=self.seed,
                checkpoint_dir=str(Path(self.checkpoint_dir) / self.wandb_project_name),
            )

        self.trainer.fit(
            model=self.retriever,
            optimizer=self.optimizer,
            train_loader=self.train_dataloader,
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            val_loader=self.val_dataloader,
            scheduler=self.lr_scheduler,
        )

        # state = self.state
        # fabric = self.fabric
        # train_dataloader = self.train_dataloader
        # val_dataloader = self.val_dataloader

        # model = state["model"]
        # optimizer = state["optimizer"]
        # lr_scheduler = state["scheduler"]

        # self.validate(fabric, model, val_dataloader, max_iters=2)  # sanity check
        # # throughput = ThroughputMonitor(fabric, window_size=5)

        # # with torch.device("meta"):
        # #     meta_model = GPT(model.config)
        # #     x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        # #     model_fwd = lambda: meta_model(x)
        # #     model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        # #     measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        # #     fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        # #     del meta_model, x

        # # get number of devices
        # # devices = fabric.world_size
        # gradient_accumulation_iters = self.compute_gradient_accumulation_iters()
        # log_iter_interval = 1 * gradient_accumulation_iters
        # initial_iter = state["iter_num"]
        # train_iterator = CycleIterator(train_dataloader)
        # train_iterator = self.progbar_wrapper(
        #     train_iterator, total=self.max_steps#, desc=f"Epoch {self.current_epoch}"
        # )

        # running_loss = RunningMean(
        #     window=gradient_accumulation_iters, sync_on_compute=False
        # ).to(fabric.device)
        # fabric.barrier()
        # total_t0 = time.perf_counter()
        # val_loss = "n/a"

        # # restore the learning rate and scheduler state
        # # if state["iter_num"] > 0:
        # #     optimizer.load_state_dict(state["optimizer_state"])
        # #     fabric.print("Restored optimizer state")
        # #     if self.lr_scheduler is not None:
        # #         self.lr_scheduler.load_state_dict(state["lr_scheduler_state"])
        # #         fabric.print("Restored lr_scheduler state")

        # for batch in train_iterator:
        #     if state["iter_num"] >= self.max_steps:
        #         break

        #     state["iter_num"] += 1
        #     iter_t0 = time.perf_counter()

        #     is_accumulating = (
        #         state["iter_num"] % self.compute_gradient_accumulation_iters() != 0
        #     )
        #     with fabric.no_backward_sync(model, enabled=is_accumulating):
        #         forward_output = model.forward(**batch, return_loss=True)
        #         # loss = chunked_cross_entropy(logits, targets)
        #         loss = forward_output["loss"]
        #         fabric.backward(loss / self.compute_gradient_accumulation_iters())

        #     running_loss.update(loss.detach())

        #     if not is_accumulating:
        #         fabric.clip_gradients(model, optimizer, max_norm=self.gradient_clip_val)
        #         optimizer.step()
        #         optimizer.zero_grad()
        #         lr_scheduler.step()
        #         state["step_count"] += 1

        #     if state["iter_num"] % log_iter_interval == 0:
        #         loss = (
        #             running_loss.compute().item()
        #         )  # expensive device-to-host synchronization
        #         t1 = time.perf_counter()
        #         # throughput.update(
        #         #     time=(t1 - total_t0),
        #         #     flops=(measured_flops * log_iter_interval),
        #         #     batches=state["iter_num"],
        #         #     samples=(state["iter_num"] * train.micro_batch_size),
        #         #     lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
        #         # )
        #         metrics = {
        #             "loss": loss,
        #             "iter": state["iter_num"],
        #             "step": state["step_count"],
        #             "epoch": train_iterator.epoch,
        #             "iter_time": t1 - iter_t0,
        #             "remaining_time": (
        #                 (t1 - total_t0)
        #                 / (state["iter_num"] - initial_iter)
        #                 * (self.max_steps - state["iter_num"])
        #             ),
        #             # "tokens": state["iter_num"] * train.micro_batch_size * model.max_seq_length,
        #             # "total_tokens": (state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size),
        #             # "learning_rate": optimizer.param_groups[0]["lr"],
        #         }
        #         if isinstance(val_loss, float):
        #             val_loss = f"{val_loss:.3f}"
        #         fabric.print(
        #             f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
        #             f" loss train: {metrics['loss']:.3f},"
        #             f" val: {val_loss} |"
        #             f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
        #             f"{' (step)' if not is_accumulating else ''}"
        #             f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
        #         )

        #         # throughput_metrics = throughput.compute()
        #         # metrics.update(throughput_metrics)
        #         fabric.log_dict(metrics, step=state["iter_num"] - 1)

    # @staticmethod
    # @torch.no_grad()
    # def validate(
    #     fabric: L.Fabric,
    #     model: torch.nn.Module,
    #     val_dataloader: DataLoader,
    #     max_iters: int,
    # ) -> torch.Tensor:
    #     fabric.barrier()
    #     fabric.print("Validating ...")
    #     model.eval()

    #     losses = []
    #     for k, batch in enumerate(val_dataloader):
    #         if k >= max_iters:
    #             break
    #         forward_output = model.forward(**batch, return_loss=True)
    #         losses.append(forward_output["loss"])

    #     val_loss = torch.stack(losses).mean()
    #     model.train()
    #     fabric.barrier()
    #     return val_loss

    # def train_(self, *args, **kwargs):
    #     """
    #     Train the model

    #     Args:
    #         *args:
    #             Additional args
    #         **kwargs:
    #             Additional kwargs

    #     Returns:
    #         `None`
    #     """
    #     if self.log_to_wandb:
    #         # TODO: add CSV logger if wandb is not available
    #         logger.info("Instantiating Wandb Logger")
    #         # log the args to wandb
    #         # logger.info(pformat(self.wandb_kwargs))
    #         self.wandb_logger = self.configure_logger(**self.wandb_kwargs)
    #         self.experiment_path = None
    #         try:
    #             self.experiment_path = Path(self.wandb_logger.experiment.dir)
    #         except Exception as e:
    #             logger.info(f"Failed to get the experiment path: {e}")

    #     # set-up training specific callbacks
    #     self.callbacks_store = self.training_callbacks()
    #     # add the evaluation callbacks
    #     self.callbacks_store.append(
    #         self.configure_prediction_callbacks(
    #             batch_size=self.prediction_batch_size,
    #             precision=self.precision,
    #         )
    #     )
    #     # add the hard negatives callback after the evaluation callback
    #     if self.max_hard_negatives_to_mine > 0:
    #         self.callbacks_store.append(self.configure_hard_negatives_callback())

    #     self.callbacks_store.append(FreeUpIndexerVRAMCallback())

    #     # if self.trainer is None:
    #     #     logger.info("Instantiating the Trainer")
    #     #     self.trainer = L.Trainer(
    #     #         accelerator=self.accelerator,
    #     #         devices=self.devices,
    #     #         num_nodes=self.num_nodes,
    #     #         strategy=self.strategy,
    #     #         accumulate_grad_batches=self.accumulate_grad_batches,
    #     #         max_epochs=self.max_epochs,
    #     #         max_steps=self.max_steps,
    #     #         gradient_clip_val=self.gradient_clip_val,
    #     #         val_check_interval=self.val_check_interval,
    #     #         check_val_every_n_epoch=self.check_val_every_n_epoch,
    #     #         deterministic=self.deterministic,
    #     #         fast_dev_run=self.fast_dev_run,
    #     #         precision=PRECISION_INPUT_STR_ALIAS_CONVERSION.get(
    #     #             self.precision, self.precision
    #     #         ),
    #     #         callbacks=self.callbacks_store,
    #     #         logger=self.wandb_logger,
    #     #         **self.trainer_kwargs,
    #     #     )
    #     # if self.fabric is None:
    #     #     self.fabric = L.Fabric(
    #     #         devices=self.devices,
    #     #         strategy=self.strategy,
    #     #         precision=PRECISION_INPUT_STR_ALIAS_CONVERSION.get(
    #     #             self.precision, self.precision
    #     #         ),
    #     #         loggers=[self.wandb_logger],
    #     #     )
    #     #     self.fabric.launch()

    #     #     self.fabric.print(pprint.pformat(self.hparams))
    #     #     if isinstance(self.wandb_logger, WandbLogger):
    #     #         self.fabric.logger.log_hyperparams(self.hparams)

    #     # # save this class as config to file
    #     # if self.experiment_path is not None:
    #     #     logger.info("Saving the configuration to file")
    #     #     self.experiment_path.mkdir(parents=True, exist_ok=True)
    #     #     OmegaConf.save(
    #     #         OmegaConf.create(to_config(self)),
    #     #         self.experiment_path / "trainer_config.yaml",
    #     #     )
    #     self.trainer.fit(
    #         self.lightning_module,
    #         datamodule=self.lightning_datamodule,
    #         ckpt_path=self.resume_from_checkpoint_path,
    #     )

    def test(
        self,
        lightning_module: GoldenRetrieverPLModule | None = None,
        checkpoint_path: str | os.PathLike | None = None,
        lightning_datamodule: GoldenRetrieverPLDataModule | None = None,
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
        if self.test_dataset is None:
            logger.warning("No test dataset provided. Skipping testing.")
            return

        if self.trainer is None:
            self.trainer = L.Trainer(
                accelerator=self.accelerator,
                devices=self.devices,
                num_nodes=self.num_nodes,
                strategy=self.strategy,
                deterministic=self.deterministic,
                fast_dev_run=self.fast_dev_run,
                precision=self.precision,
                callbacks=[
                    self.configure_prediction_callbacks(
                        batch_size=self.prediction_batch_size,
                        precision=self.precision,
                        force_reindex=force_reindex,
                    )
                ],
                **self.trainer_kwargs,
            )
        if lightning_module is not None:
            best_lightning_module = lightning_module
        else:
            try:
                if self.fast_dev_run:
                    best_lightning_module = self.lightning_module
                else:
                    # load best model for testing
                    if checkpoint_path is not None:
                        best_model_path = checkpoint_path
                    elif self.checkpoint_path is not None:
                        best_model_path = self.checkpoint_path
                    elif self.model_checkpoint_callback:
                        best_model_path = self.model_checkpoint_callback.best_model_path
                    else:
                        raise ValueError(
                            "Either `checkpoint_path` or `model_checkpoint_callback` should "
                            "be provided to the trainer"
                        )
                    logger.info(f"Loading best model from {best_model_path}")

                    best_lightning_module = (
                        GoldenRetrieverPLModule.load_from_checkpoint(best_model_path)
                    )
            except Exception as e:
                logger.info(f"Failed to load the model from checkpoint: {e}")
                logger.info("Using last model instead")
                best_lightning_module = self.lightning_module

        lightning_datamodule = lightning_datamodule or self.lightning_datamodule
        # module test
        self.trainer.test(best_lightning_module, datamodule=lightning_datamodule)

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

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
            "trainer_kwargs": to_config(cls.trainer_kwargs),
        }


def train(conf: omegaconf.DictConfig) -> None:
    logger.info("Starting training with config:")
    logger.info(pformat(OmegaConf.to_container(conf)))

    logger.info("Instantiating the Retriever")
    retriever: GoldenRetriever = hydra.utils.instantiate(
        conf.retriever, _recursive_=False
    )

    logger.info("Instantiating datasets")
    train_dataset: GoldenRetrieverDataset = hydra.utils.instantiate(
        conf.data.train_dataset, _recursive_=False
    )
    val_dataset: GoldenRetrieverDataset = hydra.utils.instantiate(
        conf.data.val_dataset, _recursive_=False
    )
    test_dataset: GoldenRetrieverDataset = hydra.utils.instantiate(
        conf.data.test_dataset, _recursive_=False
    )

    logger.info("Loading the document index")
    document_index: BaseDocumentIndex = hydra.utils.instantiate(
        conf.data.document_index, _recursive_=False
    )
    retriever.document_index = document_index

    logger.info("Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(
        conf.train,
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        _recursive_=False,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Starting testing")
    trainer.test()

    logger.info("Training and testing completed")


@hydra.main(config_path="../../conf", config_name="default", version_base="1.3")
def main(conf: omegaconf.DictConfig):
    train(conf)


def train_hydra(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    L.seed_everything(conf.train.seed)
    torch.set_float32_matmul_precision(conf.train.float32_matmul_precision)

    logger.info(f"Starting training for [bold cyan]{conf.model_name}[/bold cyan] model")
    if conf.train.pl_trainer.fast_dev_run:
        logger.info(
            f"Debug mode {conf.train.pl_trainer.fast_dev_run}. Forcing debugger configuration"
        )
        # Debuggers don't like GPUs nor multiprocessing
        # conf.train.pl_trainer.accelerator = "cpu"
        conf.train.pl_trainer.devices = 1
        conf.train.pl_trainer.strategy = "auto"
        conf.train.pl_trainer.precision = 32
        if "num_workers" in conf.data.datamodule:
            conf.data.datamodule.num_workers = {
                k: 0 for k in conf.data.datamodule.num_workers
            }
        # Switch wandb to offline mode to prevent online logging
        conf.logging.log = None
        # remove model checkpoint callback
        conf.train.model_checkpoint_callback = None

    if "print_config" in conf and conf.print_config:
        # pprint(OmegaConf.to_container(conf), console=logger, expand_all=True)
        logger.info(pformat(OmegaConf.to_container(conf)))

    # data module declaration
    logger.info("Instantiating the Data Module")
    pl_data_module: GoldenRetrieverPLDataModule = hydra.utils.instantiate(
        conf.data.datamodule, _recursive_=False
    )
    # force setup to get labels initialized for the model
    pl_data_module.prepare_data()
    # main module declaration
    pl_module: GoldenRetrieverPLModule | None = None

    if not conf.train.only_test:
        pl_data_module.setup("fit")

        # count the number of training steps
        if (
            "max_epochs" in conf.train.pl_trainer
            and conf.train.pl_trainer.max_epochs > 0
        ):
            num_training_steps = (
                len(pl_data_module.train_dataloader())
                * conf.train.pl_trainer.max_epochs
            )
            if "max_steps" in conf.train.pl_trainer:
                logger.info(
                    "Both `max_epochs` and `max_steps` are specified in the trainer configuration. "
                    "Will use `max_epochs` for the number of training steps"
                )
                conf.train.pl_trainer.max_steps = None
        elif (
            "max_steps" in conf.train.pl_trainer and conf.train.pl_trainer.max_steps > 0
        ):
            num_training_steps = conf.train.pl_trainer.max_steps
            conf.train.pl_trainer.max_epochs = None
        else:
            raise ValueError(
                "Either `max_epochs` or `max_steps` should be specified in the trainer configuration"
            )
        logger.info(f"Expected number of training steps: {num_training_steps}")

        if "lr_scheduler" in conf.model.pl_module and conf.model.pl_module.lr_scheduler:
            # set the number of warmup steps as x% of the total number of training steps
            if conf.model.pl_module.lr_scheduler.num_warmup_steps is None:
                if (
                    "warmup_steps_ratio" in conf.model.pl_module
                    and conf.model.pl_module.warmup_steps_ratio is not None
                ):
                    conf.model.pl_module.lr_scheduler.num_warmup_steps = int(
                        conf.model.pl_module.lr_scheduler.num_training_steps
                        * conf.model.pl_module.warmup_steps_ratio
                    )
                else:
                    conf.model.pl_module.lr_scheduler.num_warmup_steps = 0
            logger.info(
                f"Number of warmup steps: {conf.model.pl_module.lr_scheduler.num_warmup_steps}"
            )

        logger.info("Instantiating the Model")
        pl_module: GoldenRetrieverPLModule = hydra.utils.instantiate(
            conf.model.pl_module, _recursive_=False
        )
        if (
            "pretrain_ckpt_path" in conf.train
            and conf.train.pretrain_ckpt_path is not None
        ):
            logger.info(
                f"Loading pretrained checkpoint from {conf.train.pretrain_ckpt_path}"
            )
            pl_module.load_state_dict(
                torch.load(conf.train.pretrain_ckpt_path)["state_dict"], strict=False
            )

        if "compile" in conf.model.pl_module and conf.model.pl_module.compile:
            try:
                pl_module = torch.compile(pl_module, backend="inductor")
            except Exception:
                logger.info(
                    "Failed to compile the model, you may need to install PyTorch 2.0"
                )

    # callbacks declaration
    callbacks_store = [ModelSummary(max_depth=2)]

    experiment_logger: WandbLogger | None = None
    experiment_path: Path | None = None
    if conf.logging.log:
        logger.info("Instantiating Wandb Logger")
        experiment_logger = hydra.utils.instantiate(conf.logging.wandb_arg)
        if pl_module is not None:
            # it may happen that the model is not instantiated if we are only testing
            # in that case, we don't need to watch the model
            experiment_logger.watch(pl_module, **conf.logging.watch)
        experiment_path = Path(experiment_logger.experiment.dir)
        # Store the YaML config separately into the wandb dir
        yaml_conf: str = OmegaConf.to_yaml(cfg=conf)
        (experiment_path / "hparams.yaml").write_text(yaml_conf)
        # Add a Learning Rate Monitor callback to log the learning rate
        callbacks_store.append(LearningRateMonitor(logging_interval="step"))

    early_stopping_callback: EarlyStopping | None = None
    if conf.train.early_stopping_callback is not None:
        early_stopping_callback = hydra.utils.instantiate(
            conf.train.early_stopping_callback
        )
        callbacks_store.append(early_stopping_callback)

    model_checkpoint_callback: ModelCheckpoint | None = None
    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback,
            dirpath=experiment_path / "checkpoints" if experiment_path else None,
        )
        callbacks_store.append(model_checkpoint_callback)

    if "callbacks" in conf.train and conf.train.callbacks is not None:
        for _, callback in conf.train.callbacks.items():
            # callback can be a list of callbacks or a single callback
            if isinstance(callback, omegaconf.listconfig.ListConfig):
                for cb in callback:
                    if cb is not None:
                        callbacks_store.append(
                            hydra.utils.instantiate(cb, _recursive_=False)
                        )
            else:
                if callback is not None:
                    callbacks_store.append(hydra.utils.instantiate(callback))

    # trainer
    logger.info("Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, logger=experiment_logger
    )

    if not conf.train.only_test:
        # module fit
        trainer.fit(pl_module, datamodule=pl_data_module)

    if conf.train.pl_trainer.fast_dev_run:
        best_pl_module = pl_module
    else:
        # load best model for testing
        if conf.train.checkpoint_path:
            best_model_path = conf.evaluation.checkpoint_path
        elif model_checkpoint_callback:
            best_model_path = model_checkpoint_callback.best_model_path
        else:
            raise ValueError(
                "Either `checkpoint_path` or `model_checkpoint_callback` should "
                "be specified in the evaluation configuration"
            )
        logger.info(f"Loading best model from {best_model_path}")

        try:
            best_pl_module = GoldenRetrieverPLModule.load_from_checkpoint(
                best_model_path
            )
        except Exception as e:
            logger.info(f"Failed to load the model from checkpoint: {e}")
            logger.info("Using last model instead")
            best_pl_module = pl_module
        if "compile" in conf.model.pl_module and conf.model.pl_module.compile:
            try:
                best_pl_module = torch.compile(best_pl_module, backend="inductor")
            except Exception:
                logger.info(
                    "Failed to compile the model, you may need to install PyTorch 2.0"
                )

    # module test
    trainer.test(best_pl_module, datamodule=pl_data_module)


if __name__ == "__main__":
    main()
