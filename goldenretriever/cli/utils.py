from pathlib import Path
import sys

from omegaconf import OmegaConf
import typer
from hydra import compose, initialize, initialize_config_dir

from goldenretriever.common.log import get_logger


logger = get_logger(__name__)


def resolve_config() -> OmegaConf:
    """
    Resolve the config file and return the OmegaConf object.

    Args:
        config_path (`str`):
            The path to the config file.

    Returns:
        `OmegaConf`:
            The OmegaConf object.
    """
    # first arg is the entry point
    # second arg is the command
    # third arg is the config
    # fourth arg is the overrides
    _, _, config_path, *overrides = sys.argv
    config_path = Path(config_path)
    # TODO: do checks
    # if not config_path.exists():
    #     raise ValueError(f"File {config_path} does not exist!")
    # get path and name
    config_path, config_name = config_path.parent, config_path.stem
    logger.debug(f"config_path: {config_path}")
    logger.debug(f"config_name: {config_name}")
    # check if config_path is absolute or relative
    if config_path.is_absolute():
        context = initialize_config_dir(config_dir=str(config_path), version_base="1.3")
    else:
        context = initialize(
            config_path=f"../../{str(config_path)}", version_base="1.3"
        )

    with context:
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg
