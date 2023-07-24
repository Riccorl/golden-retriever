import os
from pathlib import Path
from typing import List, Optional, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from rich.pretty import pprint

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
from goldenretriever.common.log import get_console_logger
from goldenretriever.data.datasets import GoldenRetrieverDataset
from goldenretriever.lightning_modules.pl_data_modules import (
    GoldenRetrieverPLDataModule,
)
from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule
from goldenretriever.retriever.golden_retriever import GoldenRetriever
from goldenretriever.retriever.indexers.base import BaseDocumentIndex

logger = get_console_logger()


class Trainer:
    def __init__(
        self,
        retriever: GoldenRetriever,
        index: BaseDocumentIndex,
        train_dataset: GoldenRetrieverDataset,
        val_dataset: Union[GoldenRetrieverDataset, list[GoldenRetrieverDataset]],
        test_dataset: Optional[
            Union[GoldenRetrieverDataset, list[GoldenRetrieverDataset]]
        ] = None,
        optimizer: str = "radamw",
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 0,
        lr_scheduler: str = "linear",
        callbacks: Optional[list] = None,
        num_workers: int = 4,
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
        precision: int = 16,
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
        wandb_offline_mode: bool = False,
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
        seed: int = 42,
        float32_matmul_precision: str = "medium",
    ):
        # reproducibility
        pl.seed_everything(seed)
        # set the precision of matmul operations
        torch.set_float32_matmul_precision(float32_matmul_precision)

        # lightning data module declaration
        if isinstance(val_dataset, GoldenRetrieverDataset):
            val_dataset = [val_dataset]
        if test_dataset is not None and isinstance(
            test_dataset, GoldenRetrieverDataset
        ):
            test_dataset = [test_dataset]

        # and to the retriever
        retriever.index = index

        self.lightining_datamodule = GoldenRetrieverPLDataModule(
            train_dataset=train_dataset,
            val_datasets=val_dataset,
            test_datasets=test_dataset,
            num_workers=num_workers,
        )

        # lightning module declaration
        self.lightining_module = GoldenRetrieverPLModule(model=retriever)

        # callbacks declaration
        self.callbacks_store = [ModelSummary(max_depth=2)]

        # metric to monitor
        if isinstance(top_ks, int):
            top_ks = [top_ks]
        # order the top_ks in descending order
        top_ks = sorted(top_ks, reverse=True)
        # get the max top_k to monitor
        top_k = top_ks[0]
        metric_to_monitor = f"validate_recall@{top_k}"
        monitor_mode = "max"

        # early stopping callback if specified
        self.early_stopping_callback: Optional[EarlyStopping] = None
        if early_stopping:
            logger.log(f"Eanbling Early Stopping, patience: {early_stopping_patience}")
            self.early_stopping_callback = EarlyStopping(
                monitor=metric_to_monitor,
                mode=monitor_mode,
                patience=early_stopping_patience,
            )
            self.callbacks_store.append(self.early_stopping_callback)

        # wandb logger if specified
        self.wandb_logger: Optional[WandbLogger] = None
        self.experiment_path: Optional[Path] = None
        if log_to_wandb:
            # define some default values for the wandb logger
            if wandb_project_name is None:
                wandb_project_name = "goldenretriever"
            if wandb_save_dir is None:
                wandb_save_dir = "./"
            logger.log(f"Instantiating Wandb Logger")
            self.wandb_logger = WandbLogger(
                entity=wandb_entity,
                project=wandb_project_name,
                name=wandb_experiment_name,
                save_dir=wandb_save_dir,
                log_model=wandb_log_model,
                mode="offline" if wandb_offline_mode else "online",
            )
            self.wandb_logger.watch(self.lightining_module, log=wandb_watch)
            experiment_path = Path(self.wandb_logger.experiment.dir)
            # Store the YaML config separately into the wandb dir
            # yaml_conf: str = OmegaConf.to_yaml(cfg=conf)
            # (experiment_path / "hparams.yaml").write_text(yaml_conf)
            # Add a Learning Rate Monitor callback to log the learning rate
            self.callbacks_store.append(LearningRateMonitor(logging_interval="step"))

        # model checkpoint callback if specified
        self.model_checkpoint_callback: Optional[ModelCheckpoint] = None
        if model_checkpointing:
            logger.log(f"Enabling Model Checkpointing")
            if chekpoint_dir is None:
                chekpoint_dir = (
                    experiment_path / "checkpoints" if experiment_path else None
                )
            if checkpoint_filename is None:
                checkpoint_filename = (
                    "checkpoint-validate_recall@"
                    + top_k
                    + "_{validate_recall@"
                    + top_k
                    + ":.4f}-epoch_{epoch:02d}"
                )
            self.model_checkpoint_callback = ModelCheckpoint(
                monitor=metric_to_monitor,
                mode=monitor_mode,
                verbose=True,
                save_top_k=save_top_k,
                save_last=save_last,
                filename=checkpoint_filename,
                dirpath=chekpoint_dir,
                auto_insert_metric_name=False,
            )
            self.callbacks_store.append(self.model_checkpoint_callback)

        # prediction callback
        other_callbacks_for_prediction = [
            RecallAtKEvaluationCallback(k) for k in top_ks
        ]
        other_callbacks_for_prediction += [
            AvgRankingEvaluationCallback(k=top_k, verbose=True, prefix="train"),
            SavePredictionsCallback(),
        ]
        self.prediction_callback = GoldenRetrieverPredictionCallback(
            k=top_k,
            batch_size=prediction_batch_size,
            precision=16,
            other_callbacks=other_callbacks_for_prediction,
        )
        self.callbacks_store.append(self.prediction_callback)

        # hard negative mining callback
        self.hard_negatives_callback: Optional[NegativeAugmentationCallback] = None
        if max_hard_negatives_to_mine > 0:
            metrics_to_monitor = (
                metrics_to_monitor_for_hard_negatives or f"validate_recall@{top_k}"
            )
            self.hard_negatives_callback = NegativeAugmentationCallback(
                k=top_k,
                batch_size=prediction_batch_size,
                precision=16,
                stages=["validate"],
                metrics_to_monitor=metrics_to_monitor,
                threshold=hard_negatives_threshold,
                max_negatives=max_hard_negatives_to_mine,
                add_with_probability=mine_hard_negatives_with_probability,
                refresh_every_n_epochs=1,
                other_callbacks=[
                    AvgRankingEvaluationCallback(k=top_k, verbose=True, prefix="train")
                ],
            )
            self.callbacks_store.append(self.hard_negatives_callback)

        # utils callback
        self.callbacks_store.append(
            SaveRetrieverCallback(), FreeUpIndexerVRAMCallback()
        )

    def train(self):
        self.pl_datamodule.setup("fit")

        # count the number of training steps
        if max_epochs is not None and max_epochs > 0:
            num_training_steps = len(self.pl_datamodule.train_dataloader()) * max_epochs
            if max_steps is not None and max_steps > 0:
                logger.log(
                    f"Both `max_epochs` and `max_steps` are specified in the trainer configuration. "
                    f"Will use `max_epochs` for the number of training steps"
                )
                max_steps = None
        elif max_steps is not None and max_steps > 0:
            num_training_steps = max_steps
            max_epochs = None
        else:
            raise ValueError(
                "Either `max_epochs` or `max_steps` should be specified in the trainer configuration"
            )
        logger.log(f"Expected number of training steps: {num_training_steps}")

        if self.lr_scheduler:
            # set the number of warmup steps as x% of the total number of training steps
            if self.lr_scheduler.num_warmup_steps is None:
                if warmup_steps_ratio is not None:
                    self.lr_scheduler.num_warmup_steps = int(
                        self.lr_scheduler.num_training_steps * warmup_steps_ratio
                    )
                else:
                    self.lr_scheduler.num_warmup_steps = 0
            logger.log(f"Number of warmup steps: {self.lr_scheduler.num_warmup_steps}")

        logger.log(f"Instantiating the Model")
        if pl_module is None and self.pl_module is None:
            raise ValueError(
                "Either `pl_module` or `self.pl_module` should be provided"
            )
        pl_module: GoldenRetrieverPLModule = pl_module or self.pl_module
        if pretrain_ckpt_path is not None:
            logger.log(f"Loading pretrained checkpoint from {pretrain_ckpt_path}")
            pl_module.load_state_dict(torch.load(pretrain_ckpt_path)["state_dict"])

        if compile:
            try:
                pl_module = torch.compile(pl_module, backend="inductor")
            except Exception as e:
                logger.log(
                    f"Failed to compile the model, you may need to install PyTorch 2.0"
                )

        experiment_logger: Optional[WandbLogger] = None
        experiment_path: Optional[Path] = None
        if conf.logging.log:
            logger.log(f"Instantiating Wandb Logger")
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

    def test():
        pass


def train(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    pl.seed_everything(conf.train.seed)
    torch.set_float32_matmul_precision(conf.train.float32_matmul_precision)

    logger.log(f"Starting training for [bold cyan]{conf.model_name}[/bold cyan] model")
    if conf.train.pl_trainer.fast_dev_run:
        logger.log(
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
        pprint(OmegaConf.to_container(conf), console=logger, expand_all=True)

    # data module declaration
    logger.log(f"Instantiating the Data Module")
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
                logger.log(
                    f"Both `max_epochs` and `max_steps` are specified in the trainer configuration. "
                    f"Will use `max_epochs` for the number of training steps"
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
        logger.log(f"Expected number of training steps: {num_training_steps}")

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
            logger.log(
                f"Number of warmup steps: {conf.model.pl_module.lr_scheduler.num_warmup_steps}"
            )

        logger.log(f"Instantiating the Model")
        pl_module: GoldenRetrieverPLModule = hydra.utils.instantiate(
            conf.model.pl_module, _recursive_=False
        )
        if (
            "pretrain_ckpt_path" in conf.train
            and conf.train.pretrain_ckpt_path is not None
        ):
            logger.log(
                f"Loading pretrained checkpoint from {conf.train.pretrain_ckpt_path}"
            )
            pl_module.load_state_dict(
                torch.load(conf.train.pretrain_ckpt_path)["state_dict"], strict=False
            )

        if "compile" in conf.model.pl_module and conf.model.pl_module.compile:
            try:
                pl_module = torch.compile(pl_module, backend="inductor")
            except Exception as e:
                logger.log(
                    f"Failed to compile the model, you may need to install PyTorch 2.0"
                )

    # callbacks declaration
    callbacks_store = [ModelSummary(max_depth=2)]

    experiment_logger: Optional[WandbLogger] = None
    experiment_path: Optional[Path] = None
    if conf.logging.log:
        logger.log(f"Instantiating Wandb Logger")
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
                    callbacks_store.append(
                        hydra.utils.instantiate(cb, _recursive_=False)
                    )
            else:
                callbacks_store.append(hydra.utils.instantiate(callback))

    # trainer
    logger.log(f"Instantiating the Trainer")
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
        logger.log(f"Loading best model from {best_model_path}")

        try:
            best_pl_module = GoldenRetrieverPLModule.load_from_checkpoint(
                best_model_path
            )
        except Exception as e:
            logger.log(f"Failed to load the model from checkpoint: {e}")
            logger.log(f"Using last model instead")
            best_pl_module = pl_module
        if "compile" in conf.model.pl_module and conf.model.pl_module.compile:
            try:
                best_pl_module = torch.compile(best_pl_module, backend="inductor")
            except Exception as e:
                logger.log(
                    f"Failed to compile the model, you may need to install PyTorch 2.0"
                )

    # module test
    trainer.test(best_pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../../conf", config_name="default")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
