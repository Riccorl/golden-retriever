from typing import Any, Dict

from composer import Logger, State

from goldenretriever.common.log import get_logger

log = get_logger()


class NLPTemplateCallback:
    def __call__(
        self,
        state: State,
        logger: Logger,
        predictions: Dict,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError
