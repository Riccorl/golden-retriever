import os
from pathlib import Path
from typing import Optional

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
)
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar,
    RichProgressBarTheme,
)
from pytorch_lightning.loggers import WandbLogger
from rich.pretty import pprint

from data.pl_data_modules import PLDataModule
from models.pl_modules import GoldenRetrieverPLModule
from utils.logging import get_console_logger

logger = get_console_logger()


def train(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    pl.seed_everything(conf.train.seed)
    torch.set_float32_matmul_precision(conf.train.float32_matmul_precision)
    if conf.train.set_determinism_the_old_way:
        set_determinism_the_old_way(conf.train.pl_trainer.deterministic)
        conf.train.pl_trainer.deterministic = False

    logger.log(f"Starting training for [bold cyan]{conf.model_name}[/bold cyan] model")
    if conf.train.pl_trainer.fast_dev_run:
        logger.log(
            f"Debug mode {conf.train.pl_trainer.fast_dev_run}. Forcing debugger configuration"
        )
        # Debuggers don't like GPUs nor multiprocessing
        # conf.train.pl_trainer.accelerator = "cpu"
        conf.train.pl_trainer.devices = 1
        conf.train.pl_trainer.strategy = None
        conf.train.pl_trainer.precision = 32
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
    pl_data_module: PLDataModule = hydra.utils.instantiate(
        conf.data.datamodule, _recursive_=False
    )
    # force setup to get labels initialized for the model
    pl_data_module.prepare_data()
    # main module declaration
    pl_module: GoldenRetrieverPLModule = None

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
        else:
            raise ValueError(
                "Either `max_epochs` or `max_steps` should be specified in the trainer configuration"
            )
        logger.log(f"Expected number of training steps: {num_training_steps}")

        if "lr_scheduler" in conf.model.pl_module and conf.model.pl_module.lr_scheduler:
            conf.model.pl_module.lr_scheduler.num_training_steps = num_training_steps
            # set the number of warmup steps as 10% of the total number of training steps
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
                f"Number of warmup steps: {conf.model.pl_module.lr_scheduler.num_training_steps}"
            )

        logger.log(f"Instantiating the Model")
        pl_module: GoldenRetrieverPLModule = hydra.utils.instantiate(
            conf.model.pl_module, _recursive_=False
        )

    # callbacks declaration
    callbacks_store = [
        RichProgressBar(
            theme=RichProgressBarTheme(
                progress_bar="#802433",
                progress_bar_finished="#802433",
                progress_bar_pulse="#802433",
            )
        ),
    ]

    experiment_logger: Optional[WandbLogger] = None
    experiment_path: Optional[Path] = None
    if conf.logging.log:
        logger.log(f"Instantiating Wandb Logger")
        experiment_logger = hydra.utils.instantiate(conf.logging.wandb_arg)
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

    if (
        "prediction_callback" in conf.train
        and conf.train.prediction_callback is not None
    ):
        # for callback in conf.train.other_callbacks:
        callbacks_store.append(hydra.utils.instantiate(conf.train.prediction_callback))

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
        if conf.evaluation.checkpoint_path:
            best_model_path = conf.evaluation.checkpoint_path
        elif model_checkpoint_callback:
            best_model_path = model_checkpoint_callback.best_model_path
        else:
            raise ValueError(
                "Either `checkpoint_path` or `model_checkpoint_callback` should be specified in the evaluation configuration"
            )
        logger.log(f"Loading best model from {best_model_path}")
        best_pl_module = GoldenRetrieverPLModule.load_from_checkpoint(best_model_path)

    # module test
    trainer.test(best_pl_module, datamodule=pl_data_module)


def set_determinism_the_old_way(deterministic: bool):
    # determinism for cudnn
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)


@hydra.main(config_path="../../conf", config_name="default")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
