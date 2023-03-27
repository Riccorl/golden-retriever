import json
from pathlib import Path
from typing import Union, Any, Optional

# name constants
WEIGHTS_NAME = "weights.pt"
ONNX_WEIGHTS_NAME = "weights.onnx"
CONFIG_NAME = "config.yaml"
LABELS_NAME = "labels.json"


def load_json(path: Union[str, Path]) -> Any:
    """
    Load a json file provided in input.

    Args:
        path (`Union[str, Path]`): The path to the json file to load.

    Returns:
        `Any`: The loaded json file.
    """
    with open(path, encoding="utf8") as f:
        return json.load(f)


def dump_json(document: Any, path: Union[str, Path], indent: Optional[int] = None):
    """
    Dump input to json file.

    Args:
        document (`Any`): The document to dump.
        path (`Union[str, Path]`): The path to dump the document to.
        indent (`Optional[int]`): The indent to use for the json file.

    """
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(document, outfile, indent=indent)
