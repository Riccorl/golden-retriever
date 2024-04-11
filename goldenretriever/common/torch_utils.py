import contextlib
import platform

import psutil
import torch
from torch.utils.data import DataLoader


def get_autocast_context(
    device: str | torch.device, precision: str
) -> contextlib.AbstractContextManager:
    # fucking autocast only wants pure strings like 'cpu' or 'cuda'
    # we need to convert the model device to that
    device_type_for_autocast = str(device).split(":")[0]

    from goldenretriever.trainer import PRECISION_MAP

    # autocast doesn't work with CPU and stuff different from bfloat16
    autocast_manager = (
        contextlib.nullcontext()
        if device_type_for_autocast in ["cpu", "mps"]
        and PRECISION_MAP[precision] != torch.bfloat16
        else (
            torch.autocast(
                device_type=device_type_for_autocast,
                dtype=PRECISION_MAP[precision],
            )
        )
    )
    return autocast_manager


def build_dataloader(
    dataset, batch_size: int, num_workers: int | None = None
) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if "linux" or "macos" in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    if batch_size is not None:
        prefetch_factor = (
            max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2
        )
    else:
        prefetch_factor = 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
