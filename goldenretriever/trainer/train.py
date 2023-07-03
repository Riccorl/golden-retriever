import os
from pathlib import Path
from typing import Optional, Union

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

from goldenretriever.data.datasets import GoldenRetrieverDataset
from goldenretriever.models.model import GoldenRetriever
from goldenretriever.common.log import get_console_logger
from goldenretriever.lightning_modules.pl_data_modules import (
    GoldenRetrieverPLDataModule,
)
from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule

logger = get_console_logger()


class Trainer:
    def __init__(
        self,
        retriever: GoldenRetriever,
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
        experiment_logger: Optional[WandbLogger] = None,
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
        checkpoint_path: Optional[Union[str, os.PathLike]] = None,
        deterministic: bool = True,
        fast_dev_run: bool = False,
        precision: int = 16,
        reload_dataloaders_every_n_epochs: int = 1,
        # checkpoint_monitor: str = "validate_recall@${train.top_k}",
        # checkpoint_mode: str = "max",
        # checkpoint_save_top_k: int = 1,
        # checkpoint_save_last: bool = False,
        # checkpoint_root_dir: Optional[Union[str, os.PathLike]] = None,
        # checkpoint_filename: Union[
        #     str, os.PathLike
        # ] = "checkpoint-validate_recall@${train.top_k}_{validate_recall@${train.top_k}:.4f}-epoch_{epoch:02d}",
        # auto_insert_metric_name: bool = False,
        model_checkpoint_callback: Optional[ModelCheckpoint] = None,
        early_stopping_callback: Optional[EarlyStopping] = None,
        wandb_logger: Optional[WandbLogger] = None,
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

        if early_stopping_callback is not None:
            self.callbacks_store.append(early_stopping_callback)

        if model_checkpoint_callback is not None:
            model_checkpoint_callback = hydra.utils.instantiate(
                conf.train.model_checkpoint_callback,
                dirpath=experiment_path / "checkpoints" if experiment_path else None,
            )
            callbacks_store.append(model_checkpoint_callback)

        

    def train(
        self,
    ):
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
                torch.load(conf.train.pretrain_ckpt_path)["state_dict"]
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
                    callbacks_store.append(hydra.utils.instantiate(cb))
            else:
                callbacks_store.append(hydra.utils.instantiate(callback))

    # trainer
    logger.log(f"Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, logger=experiment_logger
    )

    if experiment_path:
        # save labels before starting training
        model_export = experiment_path / "model_export"
        model_export.mkdir(exist_ok=True, parents=True)
        # save labels
        pl_data_module.save_labels(model_export / "labels.json")

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
