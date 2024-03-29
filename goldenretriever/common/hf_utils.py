import os
from typing import Any, Dict

from composer.utils import dist
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(
    tokenizer_name: str, tokenizer_kwargs: Dict[str, Any]
) -> PreTrainedTokenizerBase:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    signal_file_path = (
        f".node_{dist.get_node_rank()}_local_rank0_completed_tokenizer_setup"
    )

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # Make sure the tokenizer files are downloaded and cached first by local rank 0
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            pass

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get(
        "model_max_length",
        int(1e30),
    )

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if dist.get_local_rank() == 0:
            with open(signal_file_path, "wb") as f:
                f.write(b"local_rank0_completed_tokenizer_setup")

        dist.barrier()

        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

    return tokenizer
