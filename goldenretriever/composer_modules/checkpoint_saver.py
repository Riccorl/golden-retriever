# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback to save checkpoints during training."""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from composer.core import Callback, Event, State, Time, Timestamp
from composer.loggers import Logger, MLFlowLogger
from composer.utils import (
    FORMAT_NAME_WITH_DIST_AND_TIME_TABLE,
    FORMAT_NAME_WITH_DIST_TABLE,
    PartialFilePath,
    checkpoint,
    create_interval_scheduler,
    create_symlink_file,
    dist,
    ensure_folder_has_no_conflicting_files,
    format_name_with_dist,
    format_name_with_dist_and_time,
    is_model_deepspeed,
    partial_format,
)
from composer.utils.compression import get_compressor, is_compressed_pt
from composer.utils.object_store.mlflow_object_store import (
    MLFLOW_EXPERIMENT_ID_FORMAT_KEY,
    MLFLOW_RUN_ID_FORMAT_KEY,
)

from composer.callbacks.checkpoint_saver import CheckpointSaver
import torch

log = logging.getLogger(__name__)

__all__ = ["CheckpointSaver"]

_TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME = ".metadata"


class MetricCheckpointSaver(CheckpointSaver):  # noqa: D101

    def __init__(
        self,
        folder: Union[str, pathlib.Path] = "{run_name}/checkpoints",
        filename: Union[str, pathlib.Path] = "ep{epoch}-ba{batch}-rank{rank}.pt",
        remote_file_name: Optional[Union[str, pathlib.Path]] = (
            "{run_name}/checkpoints/" "ep{epoch}-ba{batch}-rank{rank}.pt"
        ),
        latest_filename: Optional[Union[str, pathlib.Path]] = "latest-rank{rank}.pt",
        latest_remote_file_name: Optional[
            Union[str, pathlib.Path]
        ] = "{run_name}/checkpoints/latest-rank{rank}.pt",
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
        *,
        overwrite: bool = False,
        num_checkpoints_to_keep: int = -1,
        weights_only: bool = False,
        ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
    ):
        folder = str(folder)
        filename = str(filename)
        remote_file_name = (
            str(remote_file_name) if remote_file_name is not None else None
        )
        latest_filename = str(latest_filename) if latest_filename is not None else None
        latest_remote_file_name = (
            str(latest_remote_file_name)
            if latest_remote_file_name is not None
            else None
        )

        # want to fail early if a required CLI tool is missing to ensure no training time is wasted
        for name in [
            filename,
            remote_file_name,
            latest_filename,
            latest_remote_file_name,
        ]:
            if name is not None and is_compressed_pt(name):
                get_compressor(name).check_exists()

        if not callable(save_interval):
            save_interval = create_interval_scheduler(save_interval)
        self.save_interval = save_interval
        self.last_checkpoint_batch: Optional[Time] = None

        self.folder = folder

        self.filename = PartialFilePath(filename.lstrip("/"), folder)
        self.latest_filename = (
            PartialFilePath(latest_filename.lstrip("/"), folder)
            if latest_filename
            else None
        )
        self.remote_file_name = (
            PartialFilePath(remote_file_name) if remote_file_name else None
        )
        self.latest_remote_file_name = (
            PartialFilePath(latest_remote_file_name)
            if latest_remote_file_name
            else None
        )

        self.overwrite = overwrite
        self.saved_checkpoints: List[str] = []
        self.all_saved_checkpoints_to_timestamp: Dict[str, Timestamp] = {}
        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.weights_only = weights_only
        self.ignore_keys = ignore_keys

        # Monitor mode to keep track of the best k models
        self.monitor = monitor
        self.best_k_models: Dict[str, torch.Tensor] = {}
        self.kth_best_model_path = ""
        self.best_model_score: Optional[torch.Tensor] = None
        self.best_model_path = ""
        self.last_model_path = ""
        # self.mode = mode
        self.__init_monitor_mode(mode)

        self.start_batch = None

    def __init_monitor_mode(self, mode: str) -> None:
        torch_inf = torch.tensor(torch.inf)
        mode_dict = {"min": (torch_inf, "min"), "max": (-torch_inf, "max")}

        if mode not in mode_dict:
            raise ValueError(
                f"`mode` can be {', '.join(mode_dict.keys())} but got {mode}"
            )

        self.kth_value, self.mode = mode_dict[mode]

    def init(self, state: State, logger: Logger) -> None:
        # If MLFlowLogger is being used, format MLFlow-specific placeholders in the save folder and paths.
        # Assumes that MLFlowLogger comes before CheckpointSaver in the list of loggers.
        for destination in logger.destinations:
            if isinstance(destination, MLFlowLogger):
                mlflow_format_kwargs = {
                    MLFLOW_EXPERIMENT_ID_FORMAT_KEY: destination._experiment_id,
                    MLFLOW_RUN_ID_FORMAT_KEY: destination._run_id,
                }
                self.folder = partial_format(self.folder, **mlflow_format_kwargs)

                self.filename.folder = self.folder
                if self.latest_filename is not None:
                    self.latest_filename.folder = self.folder

                # The remote paths have the placeholders in their filename rather than folder
                if self.remote_file_name is not None:
                    self.remote_file_name.filename = partial_format(
                        self.remote_file_name.filename,
                        **mlflow_format_kwargs,
                    )
                if self.latest_remote_file_name is not None:
                    self.latest_remote_file_name.filename = partial_format(
                        self.latest_remote_file_name.filename,
                        **mlflow_format_kwargs,
                    )

                break

        folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(folder, exist_ok=True)

    def fit_start(self, state: State, logger: Logger) -> None:
        if not self.overwrite:
            # checks that save_folder contains no files with a timestamp after the current timestamp,
            # which has potential for future conflicts.
            folder = format_name_with_dist(self.folder, state.run_name)
            ensure_folder_has_no_conflicting_files(
                folder, self.filename.filename, state.timestamp
            )

        dist.barrier()  # holds all ranks until folder check is done

        if is_model_deepspeed(state.model) and self.weights_only:
            raise NotImplementedError(
                "weights_only=True is not supported when using DeepSpeed."
            )

        self.start_batch = state.timestamp.batch

    def batch_checkpoint(self, state: State, logger: Logger):
        assert callable(self.save_interval)
        if (
            self.save_interval(state, Event.BATCH_CHECKPOINT)
            and self.last_checkpoint_batch != state.timestamp.batch
        ):
            if self.monitor is not None:
                self._save_monitor_checkpoint(
                    state,
                    logger,
                )
            else:
                self._save_checkpoint(
                    state,
                    logger,
            )

    def epoch_checkpoint(self, state: State, logger: Logger):
        assert callable(self.save_interval)
        if (
            self.save_interval(state, Event.EPOCH_CHECKPOINT)
            and self.last_checkpoint_batch != state.timestamp.batch
        ):
            if self.monitor is not None:
                self._save_monitor_checkpoint(
                    state,
                    logger,
                )
            else:
                self._save_checkpoint(
                    state,
                    logger,
            )

    def iteration_checkpoint(self, state: State, logger: Logger):
        assert callable(self.save_interval)
        if (
            self.save_interval(state, Event.ITERATION_CHECKPOINT)
            and self.last_checkpoint_batch != state.timestamp.batch
        ):
            if self.monitor is not None:
                self._save_monitor_checkpoint(
                    state,
                    logger,
                )
            else:
                self._save_checkpoint(
                    state,
                    logger,
            )

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}

        all_checkpoints = []
        for save_filename, timestamp in self.all_saved_checkpoints_to_timestamp.items():
            all_checkpoints.append((save_filename, timestamp.state_dict()))

        state_dict["all_saved_checkpoints_to_timestamp"] = all_checkpoints
        return state_dict

    def load_state_dict(self, state: Dict[str, Any]):
        if "all_saved_checkpoints_to_timestamp" in state:
            for save_filename, timestamp_state in state[
                "all_saved_checkpoints_to_timestamp"
            ]:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_checkpoints_to_timestamp[save_filename] = load_timestamp

    def check_monitor_top_k(self, current: Optional[torch.Tensor] = None) -> bool:
        if current is None:
            return False

        if self.num_checkpoints_to_keep == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.num_checkpoints_to_keep
        if less_than_k_models:
            return True

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        should_update_best_and_save = monitor_op(
            current, self.best_k_models[self.kth_best_model_path]
        )

        # If using multiple devices, make sure all processes are unanimous on the decision.
        # should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

        return should_update_best_and_save

    def _save_monitor_checkpoint(self, state: State, logger: Logger):
        self.last_checkpoint_batch = state.timestamp.batch

        current = None  # monitor_candidates.get(self.monitor)

        # get eval metric from state
        eval_metric = state.eval_metrics[self.monitor]

        k = (
            len(self.best_k_models) + 1
            if self.num_checkpoints_to_keep == -1
            else self.num_checkpoints_to_keep
        )

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"), device=current.device
            )

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        is_deepspeed = is_model_deepspeed(state.model)

        if is_deepspeed and "{rank}" not in self.filename.filename:
            raise ValueError(
                f"Save filename {self.filename.filename} must have {{rank}} for deepspeed."
            )

        # save the checkpoint to the filename
        filename_with_placeholders = self.filename.format(
            state, is_deepspeed, keep_placeholders=True
        )
        # add metric to filename in place of the value of monitor
        filename_with_placeholders = filename_with_placeholders.replace(
            f"{{{self.monitor}}}", str(eval_metric)
        )

        # save the current score
        self.current_score = current
        self.best_k_models[filename_with_placeholders] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        save_filename = checkpoint.get_save_filename(state, filename_with_placeholders)
        # Store before saving so state_dict in checkpoint has reference to latest checkpoint (itself)
        self.all_saved_checkpoints_to_timestamp[save_filename] = state.timestamp

        saved_path = checkpoint.save_checkpoint(
            state=state,
            filename=filename_with_placeholders,
            weights_only=self.weights_only,
            ignore_keys=self.ignore_keys,
        )
        log.debug(f"Checkpoint locally saved to {saved_path}")

        if not saved_path:  # not all ranks save
            return

        # metadata_local_file_path = None
        # if dist.get_global_rank() == 0 and state.fsdp_sharded_state_dict_enabled:
        #     metadata_local_file_path = format_name_with_dist_and_time(
        #         os.path.join(
        #             Path(saved_path).parent,
        #             _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME,
        #         ),
        #         state.run_name,
        #         state.timestamp,
        #     )

        if self.latest_filename is not None and self.num_checkpoints_to_keep != 0:
            symlink = self.latest_filename.format(state, is_deepspeed)
            os.makedirs(os.path.dirname(symlink), exist_ok=True)
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            # Sharded checkpoints for torch >2.0 use directories not files for load_paths
            if state.fsdp_sharded_state_dict_enabled:
                src_path = str(pathlib.Path(saved_path).parent)
            else:
                src_path = saved_path
            this_rank_saves_symlinks = (
                dist.get_global_rank() == 0 or not state.fsdp_sharded_state_dict_enabled
            )
            if this_rank_saves_symlinks:
                os.symlink(os.path.relpath(src_path, os.path.dirname(symlink)), symlink)

        # if remote file name provided, upload the checkpoint
        if self.remote_file_name is not None:
            raise NotImplementedError(
                "Saving monitor checkpoints to remote file is not supported."
            )

        self.saved_checkpoints.append(saved_path)

        if del_filepath is not None and self.num_checkpoints_to_keep >= 0:
            self._delete_checkpoint(del_filepath)

    def _save_checkpoint(self, state: State, logger: Logger):
        self.last_checkpoint_batch = state.timestamp.batch

        is_deepspeed = is_model_deepspeed(state.model)

        if is_deepspeed and "{rank}" not in self.filename.filename:
            raise ValueError(
                f"Save filename {self.filename.filename} must have {{rank}} for deepspeed."
            )

        # save the checkpoint to the filename
        filename_with_placeholders = self.filename.format(
            state, is_deepspeed, keep_placeholders=True
        )
        save_filename = checkpoint.get_save_filename(state, filename_with_placeholders)
        # Store before saving so state_dict in checkpoint has reference to latest checkpoint (itself)
        self.all_saved_checkpoints_to_timestamp[save_filename] = state.timestamp

        saved_path = checkpoint.save_checkpoint(
            state=state,
            filename=filename_with_placeholders,
            weights_only=self.weights_only,
            ignore_keys=self.ignore_keys,
        )
        log.debug(f"Checkpoint locally saved to {saved_path}")

        if not saved_path:  # not all ranks save
            return

        metadata_local_file_path = None
        if dist.get_global_rank() == 0 and state.fsdp_sharded_state_dict_enabled:
            metadata_local_file_path = format_name_with_dist_and_time(
                os.path.join(
                    Path(saved_path).parent,
                    _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME,
                ),
                state.run_name,
                state.timestamp,
            )

        if self.latest_filename is not None and self.num_checkpoints_to_keep != 0:
            symlink = self.latest_filename.format(state, is_deepspeed)
            os.makedirs(os.path.dirname(symlink), exist_ok=True)
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            # Sharded checkpoints for torch >2.0 use directories not files for load_paths
            if state.fsdp_sharded_state_dict_enabled:
                src_path = str(pathlib.Path(saved_path).parent)
            else:
                src_path = saved_path
            this_rank_saves_symlinks = (
                dist.get_global_rank() == 0 or not state.fsdp_sharded_state_dict_enabled
            )
            if this_rank_saves_symlinks:
                os.symlink(os.path.relpath(src_path, os.path.dirname(symlink)), symlink)

        # if remote file name provided, upload the checkpoint
        if self.remote_file_name is not None:
            if state.fsdp_sharded_state_dict_enabled:
                remote_file_name = self.remote_file_name.format(
                    state,
                    is_deepspeed,
                    keep_placeholders=True,
                ).lstrip("/")
                assert state.sharded_ckpt_prefix_dir is not None
                remote_prefix = state.sharded_ckpt_prefix_dir
                ckpt_filename = checkpoint._TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME
                remote_file_name = os.path.join(
                    pathlib.Path(remote_file_name).parent, remote_prefix, ckpt_filename
                )
                remote_file_name = format_name_with_dist_and_time(
                    remote_file_name, state.run_name, state.timestamp
                )
                # Upload metadata file.
                # The metadata file contains info related to which shards are saved where.
                if (
                    dist.get_global_rank() == 0
                    and state.fsdp_sharded_state_dict_enabled
                ):
                    metadata_remote_file_name = format_name_with_dist_and_time(
                        os.path.join(
                            Path(remote_file_name).parent,
                            _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME,
                        ),
                        state.run_name,
                        state.timestamp,
                    )
                    assert metadata_local_file_path is not None
                    logger.upload_file(
                        remote_file_name=metadata_remote_file_name,
                        file_path=metadata_local_file_path,
                        overwrite=self.overwrite,
                    )
            else:
                remote_file_name = self.remote_file_name.format(
                    state,
                    is_deepspeed,
                ).lstrip("/")

            log.debug(f"Uploading checkpoint to {remote_file_name}")
            try:
                logger.upload_file(
                    remote_file_name=remote_file_name,
                    file_path=saved_path,
                    overwrite=self.overwrite,
                )
            except FileExistsError as e:
                raise FileExistsError(
                    f"Uploading checkpoint failed with error: {e}. overwrite was set to {self.overwrite}. To overwrite checkpoints with Trainer, set save_overwrite to True.",
                ) from e

            # symlinks stay the same with sharded checkpointing
            if self.latest_remote_file_name is not None:
                symlink_name = (
                    self.latest_remote_file_name.format(
                        state,
                        is_deepspeed,
                    ).lstrip("/")
                    + ".symlink"
                )

                # create and upload a symlink file
                with tempfile.TemporaryDirectory() as tmpdir:
                    symlink_filename = os.path.join(tmpdir, "latest.symlink")
                    # Sharded checkpoints for torch >2.0 use directories not files for load_paths
                    if state.fsdp_sharded_state_dict_enabled:
                        src_path = str(pathlib.Path(remote_file_name).parent)
                    else:
                        src_path = remote_file_name
                    log.debug(f"Creating symlink file {symlink_filename} -> {src_path}")
                    this_rank_saves_symlinks = (
                        dist.get_global_rank() == 0
                        or not state.fsdp_sharded_state_dict_enabled
                    )
                    if this_rank_saves_symlinks:
                        create_symlink_file(src_path, symlink_filename)
                        logger.upload_file(
                            remote_file_name=symlink_name,
                            file_path=symlink_filename,
                            overwrite=True,
                        )

        self.saved_checkpoints.append(saved_path)

        if self.num_checkpoints_to_keep >= 0:
            self._rotate_checkpoints(
                sharding_enabled=state.fsdp_sharded_state_dict_enabled
            )

    def _rotate_checkpoints(self, sharding_enabled: bool = False):

        while len(self.saved_checkpoints) > self.num_checkpoints_to_keep:
            prefix_dir = None
            checkpoint_to_delete = self.saved_checkpoints.pop(0)
            prefix_dir = str(Path(checkpoint_to_delete).parent)
            if not sharding_enabled:
                os.remove(checkpoint_to_delete)
            else:
                if dist.get_global_rank() == 0:
                    shutil.rmtree(prefix_dir)

    def _delete_checkpoint(
        self, checkpoint_to_delete: str, sharding_enabled: bool = False
    ):
        prefix_dir = None
        checkpoint_to_delete = checkpoint_to_delete
        prefix_dir = str(Path(checkpoint_to_delete).parent)
        if not sharding_enabled:
            os.remove(checkpoint_to_delete)
        else:
            if dist.get_global_rank() == 0:
                shutil.rmtree(prefix_dir)
