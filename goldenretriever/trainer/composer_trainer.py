from __future__ import annotations

import datetime
import itertools
import torch.distributed
from composer import (
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
from composer.core import (
    TimeUnit,
    TrainerMode,
)
from composer.trainer._deepspeed import (
    _fix_batch_precision_for_deepspeed,
)
from composer.utils import (
    ExportFormat,
    MissingConditionalImportError,
    ObjectStore,
    Transform,
    checkpoint,
    dist,
    ensure_tuple,
    export_with_logger,
    extract_hparams,
    format_name_with_dist,
    get_composer_env_dict,
    get_device,
    get_file,
    is_xla_installed,
    map_collection,
    maybe_create_object_store_from_uri,
    maybe_create_remote_uploader_downloader_from_uri,
    model_eval_mode,
    parse_uri,
    partial_format,
    reproducibility,
)
from composer.utils.misc import is_model_deepspeed
from composer.utils.object_store.mlflow_object_store import (
    MLFLOW_EXPERIMENT_ID_FORMAT_KEY,
    MLFLOW_RUN_ID_FORMAT_KEY,
)
from torch._dynamo import OptimizedModule
from torch.cuda.amp.grad_scaler import GradScaler, _refresh_per_optimizer_state
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import Metric

from goldenretriever.common.log import get_logger

log = get_logger()


class GoldenComposerTrainer(ComposerTrainer):
    def _iter_dataloader(self, trainer_mode: TrainerMode):
        """Helper method to iterate over the dataloader.

        This method yields up to :attr:`.State.dataloader_len`` batches from the dataloader. In addition, if the
        profiler is enabled, the dataloader latency recorded via the :class:`.Marker` API.

        Args:
            trainer_mode (TrainerMode): Specifies which mode the trainer is in.
        """
        assert (
            self.state.dataloader is not None
        ), "the dataloader should be set before calling this method"

        if self.state.dataloader_len is None:
            dataloader_iter = iter(self.state.dataloader)
        else:
            dataloader_iter = itertools.islice(
                self.state.dataloader, int(self.state.dataloader_len)
            )

        current_dataloader = self.state.dataloader
        while True:
            try:
                # [BEFORE/AFTER]_DATALOADER only runs while training
                if trainer_mode == TrainerMode.TRAIN:
                    self.engine.run_event(Event.BEFORE_DATALOADER)
                    # check if we need to reload the dataloader_iter
                    if current_dataloader != self.state.dataloader:
                        current_dataloader = self.state.dataloader
                        self._spin_dataloaders_to_cur_epoch()
                        # dataloader_iter = iter(self.state.dataloader)
                        if self.state.dataloader_len is None:
                            dataloader_iter = iter(self.state.dataloader)
                        else:
                            dataloader_iter = itertools.islice(
                                self.state.dataloader, int(self.state.dataloader_len)
                            )
                    # if self.reload_dataloader():
                    #     dataloader_iter = iter(self.state.dataloader)
                batch = next(dataloader_iter)
            except StopIteration:
                # [BEFORE/AFTER]_DATALOADER only runs while training
                if trainer_mode == TrainerMode.TRAIN:
                    # Event.AFTER_DATALOADER is normally called in the train loop. However, if we
                    # encounter StopIteration, the train loop will not run. Accordingly, we need to
                    # explicitly call the engine to run marker.finish() for the dataloader marker.
                    # Otherwise, we will encounter an error at the start of the next epoch when
                    # Event.BEFORE_DATALOADER tries to start an unfinished marker.
                    self.engine.run_marker_only_event(Event.AFTER_DATALOADER)
                break
            if trainer_mode == TrainerMode.TRAIN:
                print([b for b in batch["sample_idx"]])
            yield batch

    def _train_loop(self) -> None:
        """Run training for the specified number of epochs and log results."""
        # Log training start
        log.info("Using precision %s", self.state.precision)
        self.logger.log_hyperparameters(
            {
                "enabled_algorithms/" + algo.__class__.__name__: True
                for algo in self.state.algorithms
            }
        )
        assert (
            self.state.dataloader is not None
        ), "dataloader is set in __init__() or fit()"
        assert (
            self._train_data_spec is not None
        ), "The train data spec is set in __init__() or fit()"
        assert (
            self.state.scaler is not None
        ), "scaler should have been set in __init__()"

        self.engine.run_event(Event.FIT_START)

        use_grad_scaling = self._use_grad_scaling(
            self.state.precision, self.state.scaler
        )

        if self.spin_dataloaders:
            self._spin_dataloaders_to_cur_epoch()

        if self.state.timestamp.batch_in_epoch == 0 and self._rng_state is not None:
            # Only restore the rng state here if the step in the current epoch is zero.
            reproducibility.load_rng_state(self._rng_state)
            self._rng_state = None

        self.state.model.train()
        finished_epoch_early = False
        last_wct = datetime.datetime.now()

        if self.state.max_duration is None:
            # This is essentially just a type check, as max_duration should always be
            # asserted to be not None when Trainer.fit() is called
            raise RuntimeError(
                "max_duration must be specified when initializing the Trainer"
            )

        log.debug("Starting training loop")
        while self.state.timestamp < self.state.max_duration:
            if (
                int(self.state.timestamp.epoch_in_iteration) == 0
                and int(self.state.timestamp.batch_in_epoch) == 0
            ):
                self.engine.run_event(Event.ITERATION_START)

            if int(self.state.timestamp.batch_in_epoch) == 0:
                self.engine.run_event(Event.EPOCH_START)
                self.logger.log_metrics(
                    {"time/epoch": self.state.timestamp.epoch.value}
                )

            dataloader = self.state.dataloader
            if isinstance(dataloader, DataLoader) and isinstance(
                dataloader.sampler, DistributedSampler
            ):
                dataloader.sampler.set_epoch(int(self.state.timestamp.epoch))

            # for batch_idx, self.state.batch in enumerate(
            #     self._iter_dataloader(TrainerMode.TRAIN)
            # ):
            dataload_iter = enumerate(self._iter_dataloader(TrainerMode.TRAIN))
            while True:
                # track stop iteration to break out of the loop
                # get enumerator and batch
                try:
                    batch_idx, self.state.batch = next(dataload_iter)
                except StopIteration:
                    break
                if self.state.batch is None:
                    # break out of the loop
                    break
                # Spin dataloader forward unless dataloader handles internally with dataset_resumption
                if (
                    self.spin_dataloaders
                    and "train" not in self.state.dataset_resumption
                    and batch_idx
                    < int(
                        self.state.timestamp.batch_in_epoch,
                    )
                ):
                    # Restore the RNG state immediately before the next batch is yielded from the dataloader
                    if (
                        batch_idx + 1 == int(self.state.timestamp.batch_in_epoch)
                        and self._rng_state is not None
                    ):
                        reproducibility.load_rng_state(self._rng_state)
                        self._rng_state = None
                    continue

                self.state.batch = self.state.device.batch_to_device(self.state.batch)
                self.state.batch = self._train_data_spec.device_transforms(
                    self.state.batch
                )
                rank_num_samples = self._train_data_spec.get_num_samples_in_batch(
                    self.state.batch
                )
                rank_num_tokens = self._train_data_spec.get_num_tokens_in_batch(
                    self.state.batch
                )

                if self.state.deepspeed_enabled:
                    self.state.batch = _fix_batch_precision_for_deepspeed(
                        self.state.batch, self.state.precision
                    )

                self.engine.run_event(Event.AFTER_DATALOADER)

                self.engine.run_event(Event.BATCH_START)

                # Log time values
                self.logger.log_metrics(
                    {
                        "time/batch": self.state.timestamp.batch.value,
                        "time/sample": self.state.timestamp.sample.value,
                        "time/batch_in_epoch": self.state.timestamp.batch_in_epoch.value,
                        "time/sample_in_epoch": self.state.timestamp.sample_in_epoch.value,
                    }
                )
                if rank_num_tokens > 0:
                    self.logger.log_metrics(
                        {"time/token": self.state.timestamp.token.value}
                    )
                    self.logger.log_metrics(
                        {
                            "time/token_in_epoch": self.state.timestamp.token_in_epoch.value
                        }
                    )

                total_loss_dict = self._train_batch(use_grad_scaling)

                if use_grad_scaling:
                    self.state.scaler.update()

                # total_loss_dict can be None if gradient scaling failed
                if (
                    total_loss_dict is not None
                ):  # pyright: ignore[reportUnnecessaryComparison]
                    map_collection(total_loss_dict, dist.all_reduce)
                    total_loss_dict = {
                        k: loss.cpu().item() / dist.get_world_size()
                        for k, loss in total_loss_dict.items()
                    }
                    self.state.total_loss_dict = total_loss_dict
                    self.logger.log_metrics(total_loss_dict)

                # The scheduler step.step() and compute_and_log_metrics() are going to be included in the
                # next batch's wall clock time. The time accumulation must be done here so schedulers
                # have the latest timing information

                now = datetime.datetime.now()

                batch_time = now - last_wct

                total_num_samples, total_num_tokens, batch_time = (
                    self._accumulate_time_across_ranks(
                        rank_num_samples,
                        rank_num_tokens,
                        batch_time,
                    )
                )

                # `now` is actually in the past, but want to include the time it takes to perform this reduction
                last_wct = now

                if self._scheduler_step_frequency == TimeUnit.BATCH:
                    for scheduler in self.state.schedulers:
                        scheduler.step()

                if (
                    self.state.train_metrics is not None
                ):  # pyright: ignore[reportUnnecessaryComparison]
                    self._compute_and_log_metrics(
                        dataloader_label="train",
                        metrics=self.state.train_metrics,
                    )

                self.state.previous_timestamp = self.state.timestamp
                self.state.timestamp = self.state.timestamp.to_next_batch(
                    samples=total_num_samples,
                    tokens=total_num_tokens,
                    duration=batch_time,
                )

                self.engine.run_event(Event.BATCH_END)

                # Pause the timing during evaluation
                # Evaluation time is tracked separately in state.eval_timestamp
                duration = datetime.datetime.now() - last_wct
                self._run_evaluators(Event.BATCH_END)
                last_wct = datetime.datetime.now() - duration

                self.engine.run_event(Event.BATCH_CHECKPOINT)

                if self.state.timestamp >= self.state.max_duration:
                    # If max_duration is specified in batches, samples, or tokens, and
                    # and the max_duration is reached mid-epoch, then break out of the dataloader
                    # to finish the epoch early and finish training.
                    finished_epoch_early = True
                    break

            if (
                not finished_epoch_early
                or self.state.dataloader_len == self.state.timestamp.batch_in_epoch
            ):
                # Trigger the epoch end events if the dataloader was exhausted.
                # This happens if the "break" did not trigger above, or if it
                # did (e.g. duration specified in samples/batches/tokens), but it is still
                # the end of the dataloader (i.e. next(dataloader) would raise StopIteration)
                if (
                    self.state.train_metrics is not None
                ):  # pyright: ignore[reportUnnecessaryComparison]
                    self.state.train_metrics = self._ensure_metrics_device_and_dtype(
                        self.state.train_metrics
                    )
                    self._compute_and_log_metrics(
                        dataloader_label="train",
                        metrics=self.state.train_metrics,
                    )

                if self._scheduler_step_frequency == TimeUnit.EPOCH:
                    for scheduler in self.state.schedulers:
                        scheduler.step()

                self.state.previous_timestamp = self.state.timestamp
                self.state.timestamp = self.state.timestamp.to_next_epoch()

                self.engine.run_event(Event.EPOCH_END)

                # Pause the timing during evaluation
                # Evaluation time is tracked separately in state.eval_timestamp
                duration = datetime.datetime.now() - last_wct
                self._run_evaluators(Event.EPOCH_END)
                last_wct = datetime.datetime.now() - duration

                self.engine.run_event(Event.EPOCH_CHECKPOINT)

                # Increment iteration
                if (
                    self.state._iteration_length is not None
                    and self.state.timestamp.epoch_in_iteration
                    == self.state._iteration_length
                ):
                    self.state.previous_timestamp = self.state.timestamp
                    self.state.timestamp = self.state.timestamp.to_next_iteration()
                    self.engine.run_event(Event.ITERATION_END)
                    self.engine.run_event(Event.ITERATION_CHECKPOINT)

        # Log final time values
        self.logger.log_metrics(
            {
                "time/epoch": self.state.timestamp.epoch.value,
                "time/batch": self.state.timestamp.batch.value,
                "time/sample": self.state.timestamp.sample.value,
                "time/batch_in_epoch": self.state.timestamp.batch_in_epoch.value,
                "time/sample_in_epoch": self.state.timestamp.sample_in_epoch.value,
            }
        )
        if (
            self.state.previous_timestamp is not None
            and self.state.timestamp.token.value
            - self.state.previous_timestamp.token.value
            > 0
        ):
            self.logger.log_metrics({"time/token": self.state.timestamp.token.value})
            self.logger.log_metrics(
                {"time/token_in_epoch": self.state.timestamp.token_in_epoch.value}
            )

        self.engine.run_event(Event.FIT_END)
        self._run_evaluators(Event.FIT_END)
