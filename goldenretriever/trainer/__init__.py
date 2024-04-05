from composer.core import Precision

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
    "amp": "amp",
    None: None,
}
