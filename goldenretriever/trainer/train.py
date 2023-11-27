import os
from pathlib import Path
from typing import List, Optional, Union

import hydra
import lightning as pl
import omegaconf
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from pprintpp import pformat

from goldenretriever.callbacks.evaluation_callbacks import (
    AvgRankingEvaluationCallback,
    RecallAtKEvaluationCallback,
)
from goldenretriever.callbacks.prediction_callbacks import (
    GoldenRetrieverPredictionCallback,
    NegativeAugmentationCallback,
)
from goldenretriever.callbacks.utils_callbacks import (
    FreeUpIndexerVRAMCallback,
    SavePredictionsCallback,
    SaveRetrieverCallback,
)
from goldenretriever.common.log import get_logger
from goldenretriever.data.datasets import GoldenRetrieverDataset
from goldenretriever.lightning_modules.pl_data_modules import (
    GoldenRetrieverPLDataModule,
)
from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule
from goldenretriever.pytorch_modules.loss import MultiLabelNCELoss
from goldenretriever.pytorch_modules.model import GoldenRetriever
from goldenretriever.pytorch_modules.optim import RAdamW
from goldenretriever.pytorch_modules.scheduler import LinearScheduler

logger = get_logger()


class Trainer:
    def __init__(
        self,
        retriever: GoldenRetriever,
        train_dataset: GoldenRetrieverDataset,
        val_dataset: Union[GoldenRetrieverDataset, list[GoldenRetrieverDataset]],
        test_dataset: Optional[
            Union[GoldenRetrieverDataset, list[GoldenRetrieverDataset]]
        ] = None,
        num_workers: int = 4,
        optimizer: torch.optim.Optimizer = RAdamW,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = LinearScheduler,
        num_warmup_steps: int = 0,
        loss: torch.nn.Module = MultiLabelNCELoss,
        callbacks: Optional[list] = None,
        accelerator: str = "auto",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "auto",
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 1.0,
        val_check_interval: float = 1.0,
        check_val_every_n_epoch: int = 1,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        # checkpoint_path: Optional[Union[str, os.PathLike]] = None,
        deterministic: bool = True,
        fast_dev_run: bool = False,
        precision: [int, str] = 16,
        reload_dataloaders_every_n_epochs: int = 1,
        top_ks: Union[int, List[int]] = 100,
        # early stopping parameters
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        # wandb logger parameters
        log_to_wandb: bool = True,
        wandb_entity: Optional[str] = None,
        wandb_experiment_name: Optional[str] = None,
        wandb_project_name: Optional[str] = None,
        wandb_save_dir: Optional[Union[str, os.PathLike]] = None,
        wandb_log_model: bool = True,
        wandb_online_mode: bool = False,
        wandb_watch: str = "all",
        # checkpoint parameters
        model_checkpointing: bool = True,
        chekpoint_dir: Optional[Union[str, os.PathLike]] = None,
        checkpoint_filename: Optional[Union[str, os.PathLike]] = None,
        save_top_k: int = 1,
        save_last: bool = False,
        # prediction callback parameters
        prediction_batch_size: int = 128,
        # hard negatives callback parameters
        max_hard_negatives_to_mine: int = 15,
        hard_negatives_threshold: float = 0.0,
        metrics_to_monitor_for_hard_negatives: Optional[str] = None,
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
        # self.checkpoint_path = checkpoint_path
        self.deterministic = deterministic
        self.fast_dev_run = fast_dev_run
        self.precision = precision
        self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.top_ks = top_ks
        # early stopping parameters
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        # wandb logger parameters
        self.log_to_wandb = log_to_wandb
        self.wandb_entity = wandb_entity
        self.wandb_experiment_name = wandb_experiment_name
        self.wandb_project_name = wandb_project_name
        self.wandb_save_dir = wandb_save_dir
        self.wandb_log_model = wandb_log_model
        self.wandb_online_mode = wandb_online_mode
        self.wandb_watch = wandb_watch
        # checkpoint parameters
        self.model_checkpointing = model_checkpointing
        self.chekpoint_dir = chekpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self.save_top_k = save_top_k
        self.save_last = save_last
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
        pl.seed_everything(self.seed)
        # set the precision of matmul operations
        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        # lightning data module declaration
        self.lightining_datamodule = self.configure_lightning_datamodule()

        if self.max_epochs is not None:
            logger.info(f"Number of training epochs: {self.max_epochs}")
            self.max_steps = (
                len(self.lightining_datamodule.train_dataloader()) * self.max_epochs
            )

        # optimizer declaration
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # lightning module declaration
        self.lightining_module = self.configure_lightning_module()

        # callbacks declaration
        self.callbacks_store: List[pl.Callback] = self.configure_callbacks()

        logger.info("Instantiating the Trainer")
        self.trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            num_nodes=self.num_nodes,
            strategy=self.strategy,
            accumulate_grad_batches=self.accumulate_grad_batches,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            gradient_clip_val=self.gradient_clip_val,
            val_check_interval=self.val_check_interval,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            deterministic=self.deterministic,
            fast_dev_run=self.fast_dev_run,
            precision=self.precision,
            reload_dataloaders_every_n_epochs=self.reload_dataloaders_every_n_epochs,
            callbacks=self.callbacks_store,
            logger=self.wandb_logger,
        )

    def configure_lightning_datamodule(self, *args, **kwargs):
        # lightning data module declaration
        if self.val_dataset is not None and isinstance(
            self.val_dataset, GoldenRetrieverDataset
        ):
            self.val_dataset = [self.val_dataset]
        if self.test_dataset is not None and isinstance(
            self.test_dataset, GoldenRetrieverDataset
        ):
            self.test_dataset = [self.test_dataset]

        self.lightining_datamodule = GoldenRetrieverPLDataModule(
            train_dataset=self.train_dataset,
            val_datasets=self.val_dataset,
            test_datasets=self.test_dataset,
            num_workers=self.num_workers,
            *args,
            **kwargs,
        )
        return self.lightining_datamodule

    def configure_lightning_module(self, *args, **kwargs):
        # add loss object to the retriever
        if self.retriever.loss_type is None:
            self.retriever.loss_type = self.loss()

        # lightning module declaration
        self.lightining_module = GoldenRetrieverPLModule(
            model=self.retriever,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            *args,
            **kwargs,
        )

        return self.lightining_module

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

    def configure_callbacks(self, *args, **kwargs):
        # callbacks declaration
        self.callbacks_store = self.callbacks or []
        self.callbacks_store.append(ModelSummary(max_depth=2))

        # metric to monitor
        if isinstance(self.top_ks, int):
            self.top_ks = [self.top_ks]
        # order the top_ks in descending order
        # self.top_ks = sorted(self.top_ks, reverse=True)
        # get the max top_k to monitor
        self.top_k = self.top_ks[0]
        self.metric_to_monitor = f"validate_recall@{self.top_k}"
        self.monitor_mode = "max"

        # early stopping callback if specified
        self.early_stopping_callback: Optional[EarlyStopping] = None
        if self.early_stopping:
            logger.info(
                f"Eanbling Early Stopping, patience: {self.early_stopping_patience}"
            )
            self.early_stopping_callback = EarlyStopping(
                monitor=self.metric_to_monitor,
                mode=self.monitor_mode,
                patience=self.early_stopping_patience,
            )
            self.callbacks_store.append(self.early_stopping_callback)

        # wandb logger if specified
        self.wandb_logger: Optional[WandbLogger] = None
        self.experiment_path: Optional[Path] = None
        if self.log_to_wandb:
            # define some default values for the wandb logger
            if self.wandb_project_name is None:
                self.wandb_project_name = "golden-retriever"
            if self.wandb_save_dir is None:
                self.wandb_save_dir = "./"
            logger.info("Instantiating Wandb Logger")
            self.wandb_logger = WandbLogger(
                entity=self.wandb_entity,
                project=self.wandb_project_name,
                name=self.wandb_experiment_name,
                save_dir=self.wandb_save_dir,
                log_model=self.wandb_log_model,
                mode="online" if self.wandb_online_mode else "offline",
            )
            self.wandb_logger.watch(self.lightining_module, log=self.wandb_watch)
            self.experiment_path = Path(self.wandb_logger.experiment.dir)
            # Store the YaML config separately into the wandb dir
            # yaml_conf: str = OmegaConf.to_yaml(cfg=conf)
            # (experiment_path / "hparams.yaml").write_text(yaml_conf)
            # Add a Learning Rate Monitor callback to log the learning rate
            self.callbacks_store.append(LearningRateMonitor(logging_interval="step"))

        # model checkpoint callback if specified
        self.model_checkpoint_callback: Optional[ModelCheckpoint] = None
        if self.model_checkpointing:
            logger.info("Enabling Model Checkpointing")
            if self.chekpoint_dir is None:
                self.chekpoint_dir = (
                    self.experiment_path / "checkpoints"
                    if self.experiment_path
                    else None
                )
            if self.checkpoint_filename is None:
                self.checkpoint_filename = (
                    "checkpoint-validate_recall@"
                    + str(self.top_k)
                    + "_{validate_recall@"
                    + str(self.top_k)
                    + ":.4f}-epoch_{epoch:02d}"
                )
            self.model_checkpoint_callback = ModelCheckpoint(
                monitor=self.metric_to_monitor,
                mode=self.monitor_mode,
                verbose=True,
                save_top_k=self.save_top_k,
                save_last=self.save_last,
                filename=self.checkpoint_filename,
                dirpath=self.chekpoint_dir,
                auto_insert_metric_name=False,
            )
            self.callbacks_store.append(self.model_checkpoint_callback)

        # prediction callback
        self.other_callbacks_for_prediction = [
            RecallAtKEvaluationCallback(k, verbose=True) for k in self.top_ks
        ]
        self.other_callbacks_for_prediction += [
            AvgRankingEvaluationCallback(k=self.top_k, verbose=True, prefix="train"),
            # SavePredictionsCallback(),
        ]
        self.prediction_callback = GoldenRetrieverPredictionCallback(
            k=self.top_k,
            batch_size=self.prediction_batch_size,
            precision=self.precision,
            other_callbacks=self.other_callbacks_for_prediction,
            force_reindex=False,
        )
        self.callbacks_store.append(self.prediction_callback)

        # hard negative mining callback
        self.hard_negatives_callback: Optional[NegativeAugmentationCallback] = None
        if self.max_hard_negatives_to_mine > 0:
            self.metrics_to_monitor = (
                self.metrics_to_monitor_for_hard_negatives
                or f"validate_recall@{self.top_k}"
            )
            self.hard_negatives_callback = NegativeAugmentationCallback(
                k=self.top_k,
                batch_size=self.prediction_batch_size,
                precision=self.precision,
                stages=["validate"],
                metrics_to_monitor=self.metrics_to_monitor,
                threshold=self.hard_negatives_threshold,
                max_negatives=self.max_hard_negatives_to_mine,
                add_with_probability=self.mine_hard_negatives_with_probability,
                refresh_every_n_epochs=1,
                other_callbacks=[
                    AvgRankingEvaluationCallback(
                        k=self.top_k, verbose=True, prefix="train"
                    )
                ],
            )
            self.callbacks_store.append(self.hard_negatives_callback)

        # utils callback
        # self.callbacks_store.extend(
        #     [SaveRetrieverCallback(), FreeUpIndexerVRAMCallback()]
        # )
        return self.callbacks_store

    def train(self):
        # update callbacks for training specific callbacks
        self.callbacks_store = self.callbacks or []
        self.callbacks_store.extend(
            [SaveRetrieverCallback(), FreeUpIndexerVRAMCallback()]
        )
        self.trainer.fit(self.lightining_module, datamodule=self.lightining_datamodule)

    def test(
        self,
        lightining_module: Optional[GoldenRetrieverPLModule] = None,
        checkpoint_path: Optional[Union[str, os.PathLike]] = None,
        lightining_datamodule: Optional[GoldenRetrieverPLDataModule] = None,
    ):
        if lightining_module is not None:
            self.lightining_module = lightining_module
        else:
            try:
                if self.fast_dev_run:
                    best_lightining_module = self.lightining_module
                else:
                    # load best model for testing
                    if checkpoint_path is not None:
                        best_model_path = checkpoint_path
                    elif self.checkpoint_path:
                        best_model_path = self.checkpoint_path
                    elif self.model_checkpoint_callback:
                        best_model_path = self.model_checkpoint_callback.best_model_path
                    else:
                        raise ValueError(
                            "Either `checkpoint_path` or `model_checkpoint_callback` should "
                            "be provided to the trainer"
                        )
                    logger.info(f"Loading best model from {best_model_path}")

                    best_lightining_module = (
                        GoldenRetrieverPLModule.load_from_checkpoint(best_model_path)
                    )
            except Exception as e:
                logger.info(f"Failed to load the model from checkpoint: {e}")
                logger.info("Using last model instead")
                best_lightining_module = self.lightining_module

        lightining_datamodule = lightining_datamodule or self.lightining_datamodule
        # module test
        self.trainer.test(best_lightining_module, datamodule=lightining_datamodule)


def train(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    pl.seed_everything(conf.train.seed)
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
    pl_module: Optional[GoldenRetrieverPLModule] = None

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

    experiment_logger: Optional[WandbLogger] = None
    experiment_path: Optional[Path] = None
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

    early_stopping_callback: Optional[EarlyStopping] = None
    if conf.train.early_stopping_callback is not None:
        early_stopping_callback = hydra.utils.instantiate(
            conf.train.early_stopping_callback
        )
        callbacks_store.append(early_stopping_callback)

    model_checkpoint_callback: Optional[ModelCheckpoint] = None
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


@hydra.main(config_path="../../conf", config_name="default", version_base="1.3")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
