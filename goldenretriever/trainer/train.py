import os
from copy import copy, deepcopy
from pathlib import Path
from typing import List, Literal

import hydra
import lightning as pl
import omegaconf
import torch

# from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from pprintpp import pformat
from tqdm import tqdm

import goldenretriever.common.dist_utils as dist

from goldenretriever.callbacks.base import NLPTemplateCallback
from goldenretriever.callbacks.evaluation_callbacks import (
    AvgRankingEvaluationCallback,
    RecallAtKEvaluationCallback,
)
from goldenretriever.callbacks.prediction_callbacks import (
    GoldenRetrieverPredictionCallback,
)
from goldenretriever.callbacks.training_callbacks import NegativeAugmentationCallback
from goldenretriever.callbacks.utils_callbacks import (
    FreeUpIndexerVRAMCallback,
    SavePredictionsCallback,
    SaveRetrieverCallback,
)
from goldenretriever.common.from_config import FromConfig
from goldenretriever.common.log import get_logger
from goldenretriever.common.utils import get_callable_from_string, to_config

from goldenretriever.data.datasets import GoldenRetrieverStreamingDataset
from goldenretriever.indexers.base import BaseDocumentIndex
from goldenretriever.lightning_modules.pl_data_modules import (
    GoldenRetrieverPLDataModule,
)
from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule
from goldenretriever.pytorch_modules.loss import MultiLabelNCELoss
from goldenretriever.pytorch_modules.model import GoldenRetriever
from goldenretriever.pytorch_modules.optim import RAdamW
from goldenretriever.pytorch_modules.scheduler import LinearScheduler
from goldenretriever.trainer.utils import PRECISION_INPUT_STR_ALIAS_CONVERSION


from pytorch_lightning.utilities import rank_zero_only

logger = get_logger(__name__)


class Trainer(FromConfig):
    def __init__(
        self,
        retriever: GoldenRetriever,
        document_index: BaseDocumentIndex | None = None,
        train_dataset: str | GoldenRetrieverStreamingDataset | None = None,
        train_batch_size: int = 32,
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
            | GoldenRetrieverStreamingDataset
            | list[GoldenRetrieverStreamingDataset]
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
        loss: torch.nn.Module = MultiLabelNCELoss,
        micro_batch_size: int | None = None,
        automatic_optimization: bool = True,
        callbacks: list | None = None,
        accelerator: str = "auto",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "auto",
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 1.0,
        val_check_interval: float = 1.0,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: int | None = None,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        deterministic: bool = True,
        fast_dev_run: bool = False,
        precision: int | str = 16,
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
        checkpoint_dir: str | os.PathLike | None = None,
        checkpoint_filename: str | os.PathLike | None = None,
        save_top_k: int = 1,
        save_last: bool = False,
        checkpoint_kwargs: dict | None = None,
        # prediction callback parameters
        skip_eval: bool = False,
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
        # dist.initialize_dist(dist.get_device(None))
        # put all the parameters in the class
        self.retriever = retriever
        self.document_index = document_index
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
        self.add_documents_from_dataset = add_documents_from_dataset
        # trainer parameters
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.loss = loss
        self.micro_batch_size = micro_batch_size
        self.automatic_optimization = automatic_optimization
        self.callbacks = callbacks
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.num_sanity_val_steps = num_sanity_val_steps
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.deterministic = deterministic
        self.fast_dev_run = fast_dev_run
        self.precision = precision
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
        self.skip_eval = skip_eval
        self.prediction_batch_size = prediction_batch_size
        # hard negatives callback parameters
        self.max_hard_negatives_to_mine = max_hard_negatives_to_mine
        self.hard_negatives_threshold = hard_negatives_threshold
        self.metrics_to_monitor_for_hard_negatives = (
            metrics_to_monitor_for_hard_negatives
        )
        self.mine_hard_negatives_with_probability = mine_hard_negatives_with_probability
        self.hard_negative_dataset = deepcopy(self.train_dataset)
        # other parameters
        self.seed = seed
        self.float32_matmul_precision = float32_matmul_precision

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

        # reproducibility
        pl.seed_everything(self.seed)
        # set the precision of matmul operations
        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        # check if the retriever is a DictConfig
        self.retriever = self.configure_retriever()

        # lightning data module declaration
        self.lightning_datamodule = self.configure_lightning_datamodule()

        if self.max_epochs is not None:
            logger.info(f"Number of training epochs: {self.max_epochs}")
            self.max_steps = (
                len(self.lightning_datamodule.train_dataloader()) * self.max_epochs
            )

        # if self.add_documents_from_dataset:
        #     self._add_documents_from_dataset()

        # optimizer declaration
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # lightning module declaration
        self.lightning_module = self.configure_lightning_module()

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
            lightning_module=self.lightning_module,
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
        if "top_k" in self.metric_to_monitor:
            self.metric_to_monitor = self.metric_to_monitor.format(
                top_k=self.target_top_k
            )
        else:
            logger.warning(
                "The `metric_to_monitor` does not contain the `top_k` placeholder. "
                "Please make sure to include it in the metric name."
            )

        # explicitly configure some callbacks that will be needed not only by the
        # pl.Trainer but also in this class
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
        self.callbacks_store: List[pl.Callback] = []  # self.configure_callbacks()
        # add default callbacks
        self.callbacks_store += [
            # GoldenRetrieverProgressBar(),
            ModelSummary(max_depth=2),
            LearningRateMonitor(logging_interval="step"),
        ]

        # lazy trainer declaration
        self.trainer: pl.Trainer | None = None
        # init strategy
        self.strategy = (
            hydra.utils.instantiate(self.strategy)
            if isinstance(self.strategy, DictConfig)
            else self.strategy
        )

    def configure_lightning_datamodule(self, *args, **kwargs):

        self.train_dataset_kwargs["batch_size"] = self.train_batch_size

        # lightning data module declaration
        if self.val_dataset is not None and isinstance(
            self.val_dataset,
            (GoldenRetrieverStreamingDataset, GoldenRetrieverStreamingDataset, str),
        ):
            self.val_dataset = [self.val_dataset]
            self.val_dataset_kwargs["batch_size"] = self.val_batch_size
            self.val_batch_size = [self.val_batch_size]
            self.val_dataset_kwargs = [self.val_dataset_kwargs]
        if self.test_dataset is not None and isinstance(
            self.test_dataset,
            (GoldenRetrieverStreamingDataset, GoldenRetrieverStreamingDataset, str),
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
        return self.lightning_datamodule

    def configure_retriever(self, *args, **kwargs):
        # check if the retriever is a DictConfig
        if isinstance(self.retriever, omegaconf.DictConfig):
            self.retriever = hydra.utils.instantiate(self.retriever, *args, **kwargs)

        # add loss object to the retriever
        if self.retriever.loss_type is None:
            self.retriever.loss_type = self.loss()

        return self.retriever

    def _add_documents_from_dataset(self, add_test: bool = False):
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
            automatic_optimization=self.automatic_optimization,
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
        actual_save_dir = None
        if save_dir is not None:
            actual_save_dir = Path(save_dir)
            if project is not None:
                actual_save_dir = actual_save_dir / project
            if rank_zero_only.rank == 0:
                actual_save_dir.mkdir(parents=True, exist_ok=True)

        wandb_kwargs = dict(
            name=name,
            save_dir=str(actual_save_dir),
            offline=offline,
            project=project,
            log_model=log_model and not offline,
            entity=entity,
            *args,
            **kwargs,
        )
        wandb_logger = WandbLogger(**wandb_kwargs)
        if watch is not None and lightning_module is not None:
            watch_kwargs = dict(model=lightning_module)
            if watch is not None:
                watch_kwargs["log"] = watch
            wandb_logger.watch(**watch_kwargs)

        # update the config in wandb for all the ranks
        # if rank_zero_only.rank == 0:
        # wandb_logger.experiment.config.update(wandb_kwargs)
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
        save_retriever: bool = True,
        *args,
        **kwargs,
    ) -> ModelCheckpoint:
        logger.info("Enabling Model Checkpointing")
        if dirpath is None:
            dirpath = (
                self.experiment_path / "checkpoints" if self.experiment_path else None
            )
        else:
            dirpath = (
                Path(dirpath) / self.wandb_project_name
                if self.wandb_project_name
                else Path(dirpath)
            )
            # also add the run id to the dirpath
            if self.wandb_logger is not None:
                dirpath = dirpath / self.wandb_logger.experiment.id
        if filename is None:
            filename = (
                "checkpoint-" + monitor + "_{" + monitor + ":.4f}-epoch_{epoch:02d}"
            )
        dirpath.mkdir(parents=True, exist_ok=True)
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

        # update the kwargs
        # TODO: this is bad
        # kwargs.update(
        #     dirpath=self.checkpoint_dir,
        #     filename=self.checkpoint_filename,
        # )
        # modelcheckpoint_kwargs = dict(
        #     dirpath=self.checkpoint_dir,
        #     filename=self.checkpoint_filename,
        # )
        # modelcheckpoint_kwargs.update(kwargs)
        self.model_checkpoint_callback = ModelCheckpoint(**kwargs)
        self.callbacks_store.append(self.model_checkpoint_callback)

        if save_retriever:
            self.callbacks_store.append(
                SaveRetrieverCallback(saving_dir=dirpath / "retriever")
            )
        return self.model_checkpoint_callback

    def configure_hard_negatives_callback(self):
        metrics_to_monitor = (
            self.metrics_to_monitor_for_hard_negatives or self.metric_to_monitor
        )
        hard_negatives_callback = NegativeAugmentationCallback(
            k=self.target_top_k,
            batch_size=self.prediction_batch_size,
            dataset=self.hard_negative_dataset,
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
            if rank_zero_only.rank == 0:
                self.model_checkpoint_callback = self.configure_model_checkpoint(
                    **self.checkpoint_kwargs
                )
                if self.save_last:
                    self.latest_model_checkpoint_callback = (
                        self.configure_model_checkpoint(**self.last_checkpoint_kwargs)
                    )
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
        metrics_callbacks += [
            AvgRankingEvaluationCallback(k, verbose=True) for k in self.top_k
        ]
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

        prediction_callback = GoldenRetrieverPredictionCallback(
            batch_size=batch_size,
            num_workers=self.num_workers,
            precision=precision,
            k=k,
            force_reindex=force_reindex,
            other_callbacks=metrics_callbacks,
            document_index=self.document_index,
            *args,
            **kwargs,
        )
        return prediction_callback

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
            # logger.info(pformat(self.wandb_kwargs))

            self.wandb_logger = self.configure_logger(**self.wandb_kwargs)
            self.experiment_path = None
            try:
                self.experiment_path = Path(self.wandb_logger.experiment.dir)
            except Exception as e:
                logger.info(f"Failed to get the experiment path: {e}")

        # set-up training specific callbacks
        self.training_callbacks()
        # add the evaluation callbacks
        if not self.skip_eval:
            self.callbacks_store.append(
                self.configure_prediction_callbacks(
                    batch_size=self.prediction_batch_size,
                    precision=self.precision,
                )
            )
        # add the hard negatives callback after the evaluation callback
        if self.max_hard_negatives_to_mine > 0:
            self.callbacks_store.append(self.configure_hard_negatives_callback())

        self.callbacks_store.append(FreeUpIndexerVRAMCallback())

        for callback in self.callbacks_store:
            logger.info(
                f"Adding callback: {callback.__class__.__module__}.{callback.__class__.__name__}"
            )

        if not self.automatic_optimization:
            self.accumulate_grad_batches = 1

        if self.trainer is None:
            logger.info("Instantiating the Trainer")
            self.trainer = pl.Trainer(
                accelerator=self.accelerator,
                devices=self.devices,
                num_nodes=self.num_nodes,
                strategy=self.strategy,
                accumulate_grad_batches=self.accumulate_grad_batches,
                # max_epochs=self.max_epochs,
                max_steps=self.max_steps,
                # gradient_clip_val=self.gradient_clip_val,
                val_check_interval=self.val_check_interval,
                num_sanity_val_steps=self.num_sanity_val_steps,
                check_val_every_n_epoch=self.check_val_every_n_epoch,
                deterministic=self.deterministic,
                fast_dev_run=self.fast_dev_run,
                precision=PRECISION_INPUT_STR_ALIAS_CONVERSION.get(
                    self.precision, self.precision
                ),
                callbacks=self.callbacks_store,
                logger=self.wandb_logger,
                # limit_train_batches=10,
                **self.trainer_kwargs,
            )

        # # save this class as config to file
        # if self.experiment_path is not None:
        #     logger.info("Saving the configuration to file")
        #     self.experiment_path.mkdir(parents=True, exist_ok=True)
        #     OmegaConf.save(
        #         OmegaConf.create(to_config(self)),
        #         self.experiment_path / "trainer_config.yaml",
        #     )
        self.trainer.fit(
            self.lightning_module,
            datamodule=self.lightning_datamodule,
            ckpt_path=self.resume_from_checkpoint_path,
        )

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
            self.trainer = pl.Trainer(
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

    def convert_to_yaml(self):
        return OmegaConf.to_yaml(cfg=to_config(self))
