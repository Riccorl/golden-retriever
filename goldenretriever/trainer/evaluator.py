from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from composer import DataSpec, Evaluator, Event, State, Time, TimeUnit


class GoldenRetrieverEvaluator(Evaluator):
    def __init__(
        self,
        *,
        label: str,
        dataloader: Union[DataSpec, Iterable, Dict[str, Any]],
        metric_names: Optional[List[str]] = None,
        subset_num_batches: Optional[int] = None,
        eval_interval: Optional[
            Union[int, str, Time, Callable[[State, Event], bool]]
        ] = None,
        device_eval_microbatch_size: Optional[Union[int, str]] = None,
    ):
        super().__init__(
            label=label,
            dataloader=dataloader,
            metric_names=metric_names,
            subset_num_batches=subset_num_batches,
            eval_interval=eval_interval,
            device_eval_microbatch_size=device_eval_microbatch_size,
        )
        self.actual_eval_interval = None

    @property
    def eval_interval(self):
        return self._eval_interval

    @eval_interval.setter
    def eval_interval(
        self,
        eval_interval: Optional[Union[int, str, Time, Callable[[State, Event], bool]]],
    ):
        from composer.core import ensure_time
        from composer.utils import create_interval_scheduler

        if eval_interval is not None:
            self.actual_eval_interval = ensure_time(eval_interval, TimeUnit.EPOCH)

        if eval_interval is None:
            self._eval_interval = None
        elif not callable(eval_interval):
            self._eval_interval = create_interval_scheduler(
                eval_interval,
                checkpoint_events=False,
                final_events={Event.FIT_END},
            )
        else:
            self._eval_interval = eval_interval
