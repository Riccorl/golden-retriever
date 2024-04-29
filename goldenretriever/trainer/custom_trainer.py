import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast

import hydra
import lightning as L
from omegaconf import DictConfig
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from torchmetrics import RunningMean
from tqdm import tqdm
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch.optim.lr_scheduler import LRScheduler

from goldenretriever.data.utils import HardNegativesManagerThread
from goldenretriever.lightning_modules.pl_modules import HardNegativeAlgorithm


class MyCustomTrainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        max_epochs: Optional[int] = 1,
        max_steps: Optional[int] = None,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            plugins: One or several custom plugins
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end

            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
            num_nodes=num_nodes,
        )
        if seed is not None:
            # same seed for every process to init model (FSDP)
            self.fabric.seed_everything(seed)

        self.global_step = 0
        # self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        # self.hn_algo = HardNegativeAlgorithm()
        self.hn_manager: HardNegativesManagerThread | None = None

    @staticmethod
    def compute_gradient_accumulation_iters(global_batch_size, micro_batch_size) -> int:
        devices = 1
        return global_batch_size // devices // micro_batch_size

    def fit(
        self,
        model: torch.nn.Module | DictConfig,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        global_batch_size: int,
        micro_batch_size: int,
        val_loader: (
            torch.utils.data.DataLoader | List[torch.utils.data.DataLoader] | None
        ) = None,
        ckpt_path: Optional[str] = None,
        scheduler: LRScheduler | LRSchedulerConfig | None = None,
        compile: bool = False,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """
        self.fabric.launch()

        # retriever declaration
        if isinstance(model, DictConfig):
            with self.fabric.init_module(empty_init=False):
                model = hydra.utils.instantiate(model)
        if compile:
            model = torch.compile(model)
        model, optimizer = self.fabric.setup(model, optimizer)

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader)
        if val_loader is not None:
            if isinstance(val_loader, torch.utils.data.DataLoader):
                val_loader = [val_loader]
            val_loader = [self.fabric.setup_dataloaders(vl) for vl in val_loader]

        # assemble state (current epoch and global step will be added in save)
        # train dataset state will be added in save
        state = {
            "model": model,
            "optim": optimizer,
            "train_dataloader": train_loader,
            # "train_dataset_state": train_loader.dataset.state_dict(),
        }

        if scheduler is not None:
            state["scheduler"] = scheduler


        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if (
                    self.max_epochs is not None
                    and self.current_epoch >= self.max_epochs
                ):
                    self.should_stop = True

        while not self.should_stop:
            self.train_loop(
                model,
                optimizer,
                train_loader,
                global_batch_size,
                micro_batch_size,
                limit_batches=self.limit_train_batches,
                scheduler=scheduler,
            )

            if self.should_validate:
                for vl in val_loader:
                    self.val_loop(model, vl, limit_batches=self.limit_val_batches)

                self.fabric.call(
                    "hard_negative_augmentation",
                    self.fabric,
                    model,
                    train_loader,
                    self.current_epoch,
                )

            # self.step_scheduler(
            #     scheduler, level="epoch", current_value=self.current_epoch
            # )

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.save(state)

        # reset for next fit call
        self.should_stop = False

    def train_loop(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        global_batch_size: int,
        micro_batch_size: int,
        limit_batches: Union[int, float] = float("inf"),
        scheduler: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ] = None,
        passage_batch_size: int = 400,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        self.fabric.call("on_train_epoch_start", self.fabric, self.global_step)

        # gradient_accumulation_steps = self.compute_gradient_accumulation_iters(
        #     global_batch_size, micro_batch_size
        # )

        # running_loss = RunningMean(
        #     window=gradient_accumulation_steps, sync_on_compute=False
        # ).to(self.fabric.device)
        self.fabric.barrier()

        iterable = self.progbar_wrapper(
            train_loader,
            total=min(len(train_loader), limit_batches),
            desc=f"Epoch {self.current_epoch}",
        )

        collator = train_loader.collate_fn

        passages_in_batch = {}
        batches = []
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            try:
                self.hn_manager = HardNegativesManagerThread()
                sample_idxs = batch["sample_idx"]
                i = 0
                for sample in sample_idxs:
                    if sample in self.hn_manager:
                        i += 1
                        if "mined_passages" not in sample:
                            sample["mined_passages"] = {}
                        sample["mined_passages"].update(
                            {
                                tuple(passage["input_ids"]): passage
                                for passage in self.hn_manager.get(sample)
                            }
                        )
            except TypeError:
                # a little hack to avoid the initialization of the hn_manager
                # without the tokenizer and the max_length
                # return batch
                pass

            for sample in batch:
                passages_in_batch.update(
                    {
                        tuple(passage["input_ids"]): passage
                        for passage in sample["passage"]
                    }
                )
                if "mined_passages" in sample:
                    passages_in_batch.update(
                        {
                            tuple(passage["input_ids"]): passage
                            for passage in sample["mined_passages"]
                        }
                    )

            batches.extend(batch)
            if len(passages_in_batch) < passage_batch_size:
                continue

            actual_batch = collator.collate_fn(batches)
            # now split passages into batches of micro_batch_size
            actual_batch = collator.split_batch(actual_batch, micro_batch_size)

            gradient_accumulation_steps = len(actual_batch)

            for i, ba in enumerate(actual_batch):
                ba.to(self.fabric.device)

                self.fabric.call(
                    "on_train_batch_start", self.fabric, self.global_step, ba, batch_idx
                )

                # check if optimizer should step in gradient accumulation
                # should_optim_step = self.global_step % gradient_accumulation_steps == 0
                should_optim_step = i == gradient_accumulation_steps - 1
                # should_optim_step = True
                # with self.fabric.no_backward_sync(model, enabled=should_optim_step):
                forward_output = model.forward(**ba, return_loss=True)
                loss = forward_output["loss"]
                self.fabric.backward(loss)# / gradient_accumulation_steps)
                self.fabric.log("loss", loss, step=self.global_step)

                self._current_train_return = apply_to_collection(
                    forward_output, dtype=torch.Tensor, function=lambda x: x.detach()
                )

                # running_loss.update(loss.detach())

                # if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer, 0)
                # optimizer step runs train step internally through closure
                optimizer.step()
                # optimizer.step(
                #     partial(
                #         self.training_step,
                #         model=model,
                #         batch=batch,
                #         batch_idx=batch_idx,
                #     )
                # )
                self.fabric.call("on_before_zero_grad", optimizer)
                optimizer.zero_grad()
                scheduler.scheduler.step()
                # self.step_scheduler(
                #     scheduler, level="step", current_value=self.global_step
                # )
                    # log the current learning rate for each group
                    # lr_metric_dict = {
                    #     f"lr_{i}": param_group["lr"]
                    #     for i, param_group in enumerate(optimizer.param_groups)
                    # }
                    # self.fabric.log_dict(lr_metric_dict)
                    # scheduler.step()
                # else:
                #     # gradient accumulation -> no optimizer step
                #     self.training_step(model=model, batch=batch, batch_idx=batch_idx)

                self.fabric.call(
                    "on_train_batch_end", self._current_train_return, batch, batch_idx
                )

                # this guard ensures, we only step the scheduler once per global step
                # if should_optim_step:
                #     self.step_scheduler(
                #         scheduler, level="step", current_value=self.global_step
                #     )

                # add output values to progress bar
                self._format_iterable(iterable, self._current_train_return, "train")

                # only increase global step if optimizer stepped
                self.global_step += 1 #int(should_optim_step)

                # stopping criterion on step level
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    self.should_stop = True
                    break

            # reset for next batch
            passages_in_batch = {}
            batches = []

        self.fabric.call("on_train_epoch_end")

    @torch.no_grad()
    def val_loop(
        self,
        model: torch.nn.Module,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        self.fabric.barrier()
        model.eval()
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        # if val_loader is not None and not is_overridden(
        #     "validation_step", model#_unwrap_objects(model)
        # ):
        #     L.fabric.utilities.rank_zero_warn(
        #         "Your LightningModule does not have a validation_step implemented, "
        #         "but you passed a validation dataloder. Skipping Validation."
        #     )
        #     return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(
            val_loader, total=min(len(val_loader), limit_batches), desc="Validation"
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            # out = model.validation_step(batch, batch_idx)
            forward_output = model.forward(**batch, return_loss=True)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(
                forward_output, torch.Tensor, lambda x: x.detach()
            )

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")

        self.fabric.call(
            "on_validation_epoch_end",
            self.fabric,
            model,
            val_loader,
            self.current_epoch,
        )
        self.fabric.call(
            "validation_prediction_and_metrics",
            self.fabric,
            model,
            val_loader,
            self.current_epoch,
        )

        self.fabric.call("on_validation_model_train")
        model.train()
        self.fabric.barrier()
        torch.set_grad_enabled(True)

    # def training_step(
    #     self, model: L.LightningModule, batch: Any, batch_idx: int
    # ) -> torch.Tensor:
    #     """A single training step, running forward and backward. The optimizer step is called separately, as this is
    #     given as a closure to the optimizer step.

    #     Args:
    #         model: the lightning module to train
    #         batch: the batch to run the forward on
    #         batch_idx: index of the current batch w.r.t the current epoch

    #     """
    #     outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(
    #         batch, batch_idx=batch_idx
    #     )

    #     loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

    #     self.fabric.call("on_before_backward", loss)
    #     self.fabric.backward(loss)
    #     self.fabric.call("on_after_backward")

    #     # avoid gradients in stored/accumulated values -> prevents potential OOM
    #     self._current_train_return = apply_to_collection(
    #         outputs, dtype=torch.Tensor, function=lambda x: x.detach()
    #     )

    #     return loss

    def step_scheduler(
        self,
        scheduler: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler is None:
            return

        # # wrong interval (step vs. epoch)
        if scheduler.interval != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler.frequency) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update(
                {"train_" + k: v for k, v in self._current_train_return.items()}
            )

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update(
                {"val_" + k: v for k, v in self._current_val_return.items()}
            )

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler.monitor)]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler.monitor} is invalid. Possible values are {possible_keys}."
            ) from ex
        # except Exception as ex:

        if monitor is None:
            scheduler.scheduler.step()  # type: ignore[call-arg]
        else:
            scheduler.scheduler.step(monitor)

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    def progbar_wrapper(
        self, iterable: Iterable | None = None, total: int | None = None, **kwargs: Any
    ):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        train_dataloader = state.pop("train_dataloader", None)
        if train_dataloader is not None:
            train_dataloader = _unwrap_objects(train_dataloader)
            state.update(train_dataloader_state=train_dataloader.state_dict())
        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(
            os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"),
            state,
        )

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    @staticmethod
    def _format_iterable(
        prog_bar,
        candidates: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ],
        prefix: str,
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(
                candidates, torch.Tensor, lambda x: x.item() if x.numel() == 1 else x
            )
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    if isinstance(v, float):
                        postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
