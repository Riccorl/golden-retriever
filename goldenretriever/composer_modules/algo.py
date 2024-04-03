# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core MixUp classes and functions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.loss.utils import ensure_targets_one_hot

from goldenretriever.data.utils import HardNegativesManager

log = logging.getLogger(__name__)

# __all__ = ['MixUp', 'mixup_batch']



class HardNegativeAlgo(Algorithm):

    def __init__(
        self,
        tokenizer,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hn_manager = HardNegativesManager(tokenizer, max_length=max_length)

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.BEFORE_FORWARD]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        batch = state.batch

        # update hn_manager
        self.hn_manager = HardNegativesManager(self.tokenizer, max_length=self.max_length)

        # get hard negatives
        batch
