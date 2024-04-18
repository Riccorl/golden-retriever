import hydra
import typer

from goldenretriever.cli.utils import resolve_config
from goldenretriever.common.log import get_logger
from goldenretriever.trainer.train import Trainer

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def test():
    config = resolve_config()
    trainer: Trainer = hydra.utils.instantiate(config, _recursive_=False)
    trainer.test()


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def train():
    config = resolve_config()
    trainer = Trainer(**config)
    trainer.train()


if __name__ == "__main__":
    app()
