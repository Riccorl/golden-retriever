from typing import Any, Iterable
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import _update_n
import lightning as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

PRECISION_INPUT_STR_ALIAS_CONVERSION = {
    "64": "64-true",
    "32": "32-true",
    "16": "16-mixed",
    16: "16-mixed",
    32: "32-true",
    "bf16": "bf16-mixed",
    "fp32": "32-true",
    "fp16": "16-mixed",
}


class GoldenRetrieverProgressBar(TQDMProgressBar):
    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # get current step from the tqdm bar
        n = self.train_progress_bar.n
        n += batch["questions"]["input_ids"].size(0)
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self):
        return self
