from composer.core import Precision
import torch

PRECISION_MAP = {
    None: torch.float32,
    32: torch.float32,
    16: torch.float16,
    torch.float32: torch.float32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "half": torch.float16,
    "32": torch.float32,
    "16": torch.float16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "amp_fp16": torch.float16,
    "amp_bf16": torch.bfloat16,
    "mixed": torch.float16,
    "true": torch.float32,
    "mixed-precision": torch.float16,
}

PRECISION_INPUT_STR_ALIAS_CONVERSION = {
    "64": "64-true",
    "32": "32-true",
    "16": "16-mixed",
    "bf16": "bf16-mixed",
}


COMPOSER_PRECISION_INPUT_STR_ALIAS_CONVERSION = {
    "32": Precision.FP32,
    "16": Precision.AMP_FP16,
    "bf16": Precision.AMP_BF16,
    "fp8": Precision.AMP_FP8,
    32: Precision.FP32,
    16: Precision.AMP_FP16,
    8: Precision.AMP_FP8,
    "fp32": Precision.FP32,
    "fp16": Precision.AMP_FP16,
    "amp_fp16": Precision.AMP_FP16,
    "amp_bf16": Precision.AMP_BF16,
    "amp_fp8": Precision.AMP_FP8,
    None: None,
}
