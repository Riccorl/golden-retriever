import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union

import composer
import composer.callbacks
import hydra
import lightning as pl
import omegaconf
import torch
from composer import (
    Algorithm,
    DataSpec,
    Evaluator,
    Event,
    State,
    Time,
    TimeUnit,
)
from composer import (
    Trainer as ComposerTrainer,
)
from composer.callbacks import (
    CheckpointSaver,
    EarlyStopper,
    LRMonitor,
    MemoryMonitor,
    SpeedMonitor,
)
from composer.loggers import WandBLogger
from composer.utils import dist, reproducibility

# from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

# from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from pprintpp import pformat
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from goldenretriever.common.log import _golden_retriever_build_pbar, get_logger
from goldenretriever.common.model_inputs import ModelInputs
from goldenretriever.common.utils import to_config
from goldenretriever.composer_modules.algo import HardNegativeAlgo
from goldenretriever.composer_modules.callbacks import (
    HardNegativeMiningCallback,
    PredictionCallback,
)
from goldenretriever.composer_modules.callbacks import (
    RecallAtKEvaluationCallback as ComposerRecallAtKEvaluationCallback,
)
from goldenretriever.composer_modules.checkpoint_saver import MetricCheckpointSaver
from goldenretriever.composer_modules.mosaic_module import GoldenRetrieverComposerModule
from goldenretriever.data.datasets import GoldenRetrieverDataset
from goldenretriever.data.mosaic_datasets import (
    GoldenRetrieverCollator,
    GoldenStreamingDataLoader,
    StreamingGoldenRetrieverDataset,
)
from goldenretriever.indexers.base import BaseDocumentIndex
from goldenretriever.lightning_modules.pl_data_modules import (
    GoldenRetrieverPLDataModule,
)
from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule
from goldenretriever.pytorch_modules.loss import MultiLabelNCELoss
from goldenretriever.pytorch_modules.model import GoldenRetriever
from goldenretriever.pytorch_modules.optim import RAdamW
from goldenretriever.pytorch_modules.scheduler import LinearScheduler
from goldenretriever.trainer import (
    COMPOSER_PRECISION_INPUT_STR_ALIAS_CONVERSION,
    PRECISION_INPUT_STR_ALIAS_CONVERSION,
)
from goldenretriever.trainer.composer_trainer import GoldenComposerTrainer
from goldenretriever.trainer.evaluator import GoldenRetrieverEvaluator

logger = get_logger()


class Trainer(FromConfig):
    def __init__(
        self,
        retriever: GoldenRetriever,
        train_dataset: GoldenRetrieverDataset | None = None,
        val_dataset: (
            GoldenRetrieverDataset | list[GoldenRetrieverDataset] | None
        ) = None,
        test_dataset: (
            GoldenRetrieverDataset | list[GoldenRetrieverDataset] | None
        ) = None,
        num_workers: int = 4,
        optimizer: torch.optim.Optimizer = RAdamW,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = LinearScheduler,
        num_warmup_steps: int = 0,
        loss: torch.nn.Module = MultiLabelNCELoss,
        callbacks: list | None = None,
        device: str = "gpu",
        precision: int | str = "amp",
        # accelerator: str = "auto",
        # devices: int = 1,
        # accumulate_grad_batches: int = 1,
        # gradient_clip_val: float = 1.0,
        # val_check_interval: float = 1.0,
        # check_val_every_n_epoch: int = 1,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        eval_interval: int | str | Time | Callable[[State, Event], bool] = "1ba",
        deterministic: bool = True,
        fast_dev_run: bool = False,
        resume_from_checkpoint_path: str | os.PathLike | None = None,
        deepspeed_config: dict | None = None,
        fsdp_config: dict | None = None,
        dist_timeout: int = 300,
        composer_trainer_kwargs: dict | None = None,
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
        wandb_log_model: bool = True,
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
        # put all the parameters in the class
        self.retriever = retriever
        # datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        # trainer parameters
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.loss = loss
        self.callbacks = callbacks
        self.device = device
        self.precision = precision
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.eval_interval = eval_interval
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
        if self.deterministic:
            reproducibility.configure_deterministic_mode()
        reproducibility.seed_all(self.seed)
        # set the precision of matmul operations
        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        # dataloader declaration
        self.train_dataloader, self.val_dataloader = self.configure_dataloader()

        if self.max_epochs is not None:
            logger.info(f"Number of training epochs: {self.max_epochs}")
            self.max_steps = len(self.train_dataloader) * self.max_epochs

        # optimizer declaration
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # composer module declaration
        self.composer_module = self.configure_composer_module()

        # logger and experiment declaration
        wandb_args = dict(
            project=self.wandb_project_name,
            group=self.wandb_group,
            name=self.wandb_experiment_name,
            entity=self.wandb_entity,
            tags=self.wandb_tags,
            log_artifacts=self.wandb_log_artifacts,
            rank_zero_only=self.wandb_rank_zero_only,
        )
        wandb_init_kwargs = {
            "save_dir": self.wandb_save_dir,
            # TODO: maybe env variables with "dryrun" and "online" would be better
            "mode": "online" if self.wandb_online_mode else "offline",
            "dir": self.wandb_save_dir,
        }
        if self.wandb_kwargs is not None:
            wandb_init_kwargs.update(self.wandb_kwargs)
            wandb_args.update({"init_kwargs": wandb_init_kwargs})
        self.wandb_kwargs = wandb_args
        self.wandb_logger: WandBLogger | None = None
        self.experiment_path: Path | None = None

        # setup metrics to monitor for a bunch of callbacks
        if isinstance(self.top_k, int):
            self.top_k = [self.top_k]
        # save the target top_k
        self.target_top_k = self.top_k[0]
        logger.info(
            f"Monitor top-k value is recall@{self.target_top_k}. \n"
            "If you provided a list of top-k values, the first one will be used."
        )
        self.metric_to_monitor = self.metric_to_monitor.format(top_k=self.target_top_k)

        # explicitly configure some callbacks that will be needed not only by the
        # Composer Trainer but also in this class
        checkpoint_kwargs = dict(
            folder=self.checkpoint_dir,
            filename=self.checkpoint_filename,
            save_interval=self.save_interval,
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
            last_checkpoint_kwargs["filename"] = "last-{epoch}-{step}"
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
        self.callbacks_store: List[composer.Callback] = []  # self.configure_callbacks()
        # add default callbacks
        self.callbacks_store += [
            SpeedMonitor(window_size=100),
            LRMonitor(),
            MemoryMonitor(),
        ]

        # algorithms declaration
        self.agorithms: List[Algorithm] = []

        # lazy trainer declaration
        self.trainer: ComposerTrainer | None = None

    def configure_dataloader(self, *args, **kwargs):

        train_collator = GoldenRetrieverCollator(
            tokenizer=self.retriever.question_tokenizer
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=train_collator,
            batch_size=32,
            drop_last=False,
            num_workers=self.num_workers,
            # pin_memory=True,
            # prefetch_factor=2,
            # persistent_workers=True,
            # timeout=0,
        )
        self.train_dataloader = DataSpec(
            self.train_dataloader,
            split_batch=StreamingGoldenRetrieverDataset.split_batch,
            get_num_samples_in_batch=StreamingGoldenRetrieverDataset.get_num_samples_in_batch,
            get_num_tokens_in_batch=StreamingGoldenRetrieverDataset.get_num_tokens_in_batch,
        )

        # self.val_dataloader = GoldenStreamingDataLoader(
        val_collator = GoldenRetrieverCollator(
            tokenizer=self.retriever.question_tokenizer
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            collate_fn=val_collator,
            batch_size=32,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            timeout=0,
        )
        self.val_dataloader = DataSpec(
            self.val_dataloader,
            split_batch=StreamingGoldenRetrieverDataset.split_batch,
            get_num_samples_in_batch=StreamingGoldenRetrieverDataset.get_num_samples_in_batch,
            get_num_tokens_in_batch=StreamingGoldenRetrieverDataset.get_num_tokens_in_batch,
        )
        return self.train_dataloader, self.val_dataloader

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
                self.lr_scheduler = self.lr_scheduler(
                    optimizer=self.optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.max_steps,
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
    ) -> WandBLogger:
        """
        Configure the wandb logger

        Args:


        Returns:
        """
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
        # if watch is not None and lightning_module is not None:
        #     watch_kwargs = dict(model=lightning_module)
        #     if watch is not None:
        #         watch_kwargs["log"] = watch
        #     wandb_logger.watch(**watch_kwargs)
        return wandb_logger

    @staticmethod
    def configure_early_stopping(
        monitor: str,
        dataloader_label: str,
        patience: int = 3,
        *args,
        **kwargs,
    ) -> EarlyStopping:
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
                self.experiment_path / "checkpoints" if self.experiment_path else None
            )
        if filename is None:
            filename = (
                "checkpoint-" + monitor + "_{" + monitor + ":.4f}-epoch_{epoch:02d}"
            )
        self.checkpoint_path = folder / filename if folder is not None else None
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
        self.model_checkpoint_callback = CheckpointSaver(**kwargs)
        return self.model_checkpoint_callback

    def configure_hard_negatives_callback(self):
        # TODO: use new callback and algorithm for composer
        metrics_to_monitor = (
            self.metrics_to_monitor_for_hard_negatives or self.metric_to_monitor
        )
        # hard_negatives_callback = NegativeAugmentationCallback(
        #     k=self.target_top_k,
        #     batch_size=self.prediction_batch_size,
        #     precision=self.precision,
        #     stages=["validate"],
        #     metrics_to_monitor=metrics_to_monitor,
        #     threshold=self.hard_negatives_threshold,
        #     max_negatives=self.max_hard_negatives_to_mine,
        #     add_with_probability=self.mine_hard_negatives_with_probability,
        #     refresh_every_n_epochs=1,
        # )
        hard_negatives_callback = HardNegativeMiningCallback(
            k=self.target_top_k, interval=self.eval_interval
        )
        hard_negative_algo = HardNegativeAlgo(
            self.train_dataset.tokenizer,
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

        # prediction_callback = GoldenRetrieverPredictionCallback(
        #     batch_size=batch_size,
        #     precision=precision,
        #     k=k,
        #     force_reindex=force_reindex,
        #     other_callbacks=metrics_callbacks,
        #     *args,
        #     **kwargs,
        # )

        prediction_callback = PredictionCallback(
            k=k,
            batch_size=batch_size,
            precision=precision,
            force_reindex=force_reindex,
            other_callbacks=metrics_callbacks,
            # other_callbacks=[ComposerRecallAtKEvaluationCallback()],
            interval=interval,
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
            self.experiment_path = Path(self.wandb_logger.experiment.dir)

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

        # set-up training specific callbacks
        self.callbacks_store = self.training_callbacks()

        # self.callbacks_store.append(FreeUpIndexerVRAMCallback())

        self.configure_dataloader()
        evaluators = [
            GoldenRetrieverEvaluator(
                label="val_dataset",
                dataloader=self.val_dataloader,
            )
        ]
        if self.trainer is None:
            logger.info("Instantiating the Trainer")
            self.trainer = ComposerTrainer(
                model=self.composer_module,
                train_dataloader=self.train_dataloader,
                eval_dataloader=evaluators,
                device_train_microbatch_size=self.device_train_microbatch_size,
                algorithms=self.agorithms,
                progress_bar=self.progress_bar,
                log_to_console=self.log_to_console,
                device=self.device,
                precision=COMPOSER_PRECISION_INPUT_STR_ALIAS_CONVERSION.get(
                    self.precision, self.precision
                ),
                optimizers=self.optimizer,
                schedulers=self.lr_scheduler,
                step_schedulers_every_batch=self.step_schedulers_every_batch,  # interval should be step
                max_duration="230ba",
                eval_interval=self.eval_interval,
                dist_timeout=self.dist_timeout,
                load_path=self.resume_from_checkpoint_path,
                seed=self.seed,
                callbacks=self.callbacks_store,
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
            )

            # self.trainer = pl.Trainer(
            #     accelerator=self.accelerator,
            #     devices=self.devices,
            #     num_nodes=self.num_nodes,
            #     strategy=self.strategy,
            #     accumulate_grad_batches=self.accumulate_grad_batches,
            #     max_epochs=self.max_epochs,
            #     max_steps=self.max_steps,
            #     gradient_clip_val=self.gradient_clip_val,
            #     val_check_interval=self.val_check_interval,
            #     check_val_every_n_epoch=self.check_val_every_n_epoch,
            #     deterministic=self.deterministic,
            #     fast_dev_run=self.fast_dev_run,
            #     precision=PRECISION_INPUT_STR_ALIAS_CONVERSION.get(
            #         self.precision, self.precision
            #     ),
            #     reload_dataloaders_every_n_epochs=self.reload_dataloaders_every_n_epochs,
            #     callbacks=self.callbacks_store,
            #     logger=self.wandb_logger,
            #     **self.trainer_kwargs,
            # )

        self.trainer.fit()

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
                **self.composer_trainer_kwargs,
            )
        if lightning_module is not None:
            best_lightning_module = lightning_module
        else:
            try:
                if self.fast_dev_run:
                    best_lightning_module = self.composer_module
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
                best_lightning_module = self.composer_module

        lightning_datamodule = lightning_datamodule or self.lightning_datamodule
        # module test
        self.trainer.test(best_lightning_module, datamodule=lightning_datamodule)

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
