import logging
import os
import sys
import threading
from logging.config import dictConfig
from typing import Any, Dict, Optional

from art import text2art, tprint
from colorama import Fore, Style, init
from rich import get_console


import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO, Union

import tqdm.auto
import yaml
from composer.core.time import TimeUnit
from composer.loggers.logger import Logger, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.progress_bar_logger import ProgressBarLogger, _ProgressBar, _IS_TRAIN_TO_KEYS_TO_LOG
from composer.utils import dist, is_notebook

# if TYPE_CHECKING:
from composer.core import State, Timestamp

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

_default_log_level = logging.WARNING

# fancy logger
_console = get_console()


class ColorfulFormatter(logging.Formatter):
    """
    Formatter to add coloring to log messages by log type
    """

    COLORS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.CYAN,
        # "INFO": Fore.GREEN,
    }

    def format(self, record):
        record.rank = int(os.getenv("LOCAL_RANK", "0"))
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, "") + log_message + Fore.RESET


DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
        },
        "colorful": {
            "()": ColorfulFormatter,
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] [RANK:%(rank)d] %(message)s",
        },
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "filters": [],
            "stream": sys.stdout,
        },
        "color_console": {
            "class": "logging.StreamHandler",
            "formatter": "colorful",
            "filters": [],
            "stream": sys.stdout,
        },
    },
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {
        "goldenretriever": {
            "handlers": ["color_console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}


def configure_logging():
    """Configure with default logging"""
    init()  # Initialize colorama
    dictConfig(DEFAULT_LOGGING_CONFIG)


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_default_log_level)
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def set_log_level(level: int, logger: logging.Logger = None) -> None:
    """
    Set the log level.
    Args:
        level (:obj:`int`):
            Logging level.
        logger (:obj:`logging.Logger`):
            Logger to set the log level.
    """
    if not logger:
        _configure_library_root_logger()
        logger = _get_library_root_logger()
    logger.setLevel(level)


def get_logger(
    name: str | None = None,
    level: int | None = None,
    formatter: str | None = None,
) -> logging.Logger:
    """
    Return a logger with the specified name.
    """

    configure_logging()

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()

    if level is not None:
        set_log_level(level)

    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    _default_handler.setFormatter(formatter)

    return logging.getLogger(name)


def get_console_logger():
    return _console


def print_relik_text_art(
    text: str = "golden-retriever", font: str = "larry3d", **kwargs
):
    tprint(text, font=font, **kwargs)


# class GoldenRetrieverProgressBarLogger(ProgressBarLogger):
#     """
#     We subclass the ProgressBarLogger to add a custom progress bar for the GoldenRetriever training loop.
#     """

def _golden_retriever_build_pbar(self, state: State, is_train: bool) -> _ProgressBar:
    """Builds a pbar.

    *   If ``state.max_duration.unit`` is :attr:`.TimeUnit.EPOCH`, then a new progress bar will be created for each epoch.
        Mid-epoch evaluation progress bars will be labeled with the batch and epoch number.
    *   Otherwise, one progress bar will be used for all of training. Evaluation progress bars will be labeled
        with the time (in units of ``max_duration.unit``) at which evaluation runs.
    """
    # Always using position=1 to avoid jumping progress bars
    # In jupyter notebooks, no need for the dummy pbar, so use the default position
    position = None if is_notebook() else 1
    desc = f'{state.dataloader_label:15}'
    max_duration_unit = None if state.max_duration is None else state.max_duration.unit

    if max_duration_unit == TimeUnit.EPOCH or max_duration_unit is None:
        total = int(state.dataloader_len) if state.dataloader_len is not None else None
        timestamp_key = 'sample_in_epoch'

        unit = TimeUnit.SAMPLE
        n = state.timestamp.epoch.value
        if self.train_pbar is None and not is_train:
            # epochwise eval results refer to model from previous epoch (n-1)
            n = n - 1 if n > 0 else 0
        if self.train_pbar is None:
            desc += f'Epoch {n:3}'
        else:
            # For evaluation mid-epoch, show the total batch count
            desc += f'Sample {int(state.timestamp.sample):3}'
        desc += ': '

    else:
        if is_train:
            assert state.max_duration is not None, 'max_duration should be set if training'
            unit = max_duration_unit
            total = state.max_duration.value
            # pad for the expected length of an eval pbar -- which is 14 characters (see the else logic below)
            desc = desc.ljust(len(desc) + 14)
        else:
            unit = TimeUnit.BATCH
            total = int(state.dataloader_len) if state.dataloader_len is not None else None
            value = int(state.timestamp.get(max_duration_unit))
            # Longest unit name is sample (6 characters)
            desc += f'{max_duration_unit.name.capitalize():6} {value:5}: '

        timestamp_key = unit.name.lower()

    return _ProgressBar(
        file=self.stream,
        total=total,
        position=position,
        keys_to_log=_IS_TRAIN_TO_KEYS_TO_LOG[is_train],
        # In a notebook, the `bar_format` should not include the {bar}, as otherwise
        # it would appear twice.
        bar_format=desc + ' {l_bar}' + ('' if is_notebook() else '{bar:25}') + '{r_bar}{bar:-1b}',
        unit=unit.value.lower(),
        metrics={},
        timestamp_key=timestamp_key,
    )