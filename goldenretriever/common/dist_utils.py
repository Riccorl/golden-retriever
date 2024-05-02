from typing import Any, List, Optional, Sequence, TypeVar, cast

import streaming.base.distributed as streaming_dist
import torch
import torch.distributed as torch_dist
from streaming.base.world import World
from torch._C._distributed_c10d import ProcessGroup
import composer.utils.dist as composer_dist

TObj = TypeVar("TObj")


def barrier() -> None:
    """Synchronizes all processes."""
    if torch_dist.is_available() and torch_dist.is_initialized():
        streaming_dist.barrier()
    return


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Get the rank of the current process.

    .. seealso:: :func:`torch.distributed.get_rank`

    Returns:
        int: The rank of the current process.
    """
    if torch_dist.is_available() and torch_dist.is_initialized():
        return torch_dist.get_rank(group)
    return 0


def get_local_world_size() -> int:
    """Returns the local world size, which is the number of processes for the current node.

    Returns:
        int: The local world size.
    """
    try:
        return composer_dist.get_local_world_size()
    except Exception:
        return torch.cuda.device_count()


def get_world_size() -> int:
    """Returns the world size, which is the number of processes participating in this training run.

    Returns:
        int: The world size.
    """
    return composer_dist.get_world_size()


def all_gather_object(obj: TObj, group=None) -> List[TObj]:
    """Collect a pickleable object from each rank and return a list of these objects indexed by rank.

    .. seealso:: :func:`torch.distributed.all_gather_object`

    Args:
        obj (TObj): Object to be gathered.
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.

    Returns:
        List[TObj]: A list of objects indexed by rank.
    """
    world = World.detect()
    if torch_dist.is_available() and torch_dist.is_initialized():
        obj_gather_list = [None for _ in range(world.num_ranks)]
        torch_dist.all_gather_object(obj_gather_list, obj, group=group)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return cast(List[TObj], obj_gather_list)
    world_size = world.num_ranks
    if world_size == 1:
        return [obj]
    raise RuntimeError(
        f"The world_size({world_size}) > 1, but the distributed package is not "
        "available or has not been initialized. Please check you have initialized "
        "the distributed runtime and that PyTorch has been built with distributed "
        "support. If calling this function outside Trainer, please ensure that "
        "`composer.utils.dist.initialize_dist` has been called first.",
    )


def all_gather(tensor: torch.Tensor, group=None) -> Sequence[torch.Tensor]:
    """Collects a :class:`~torch.Tensor` from each rank.

    .. seealso:: :func:`torch.distributed.all_gather`

    Args:
        tensor (torch.Tensor): Tensor from each rank to be gathered.
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.

    Returns:
        Sequence[Tensor]: A sequence of tensors indexed by rank.
    """
    world = World.detect()
    if torch_dist.is_available() and torch_dist.is_initialized():
        obj_gather_list = [torch.zeros_like(tensor) for _ in range(world.num_ranks)]
        torch_dist.all_gather(obj_gather_list, tensor, group=group)
        return obj_gather_list
    world_size = world.num_ranks
    if world_size == 1:
        return [tensor]
    raise RuntimeError(
        f"The world_size({world_size}) > 1, but the distributed package is not "
        "available or has not been initialized. Please check you have initialized "
        "the distributed runtime and that PyTorch has been built with distributed "
        "support. If calling this function outside Trainer, please ensure that "
        "`composer.utils.dist.initialize_dist` has been called first.",
    )


def broadcast(tensor: torch.Tensor, src: int) -> None:
    """Broadcasts the tensor to the whole group.

    Args:
        tensor (Tensor): Data to be sent if src is the rank of current process, and tensor to be
            used to save received data otherwise.
        src (int): Source rank.
    """
    streaming_dist.broadcast(tensor, src)


def broadcast_object_list(object_list: List[Any], src: int = 0, group=None) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be broadcasted.

    .. seealso:: :func:`torch.distributed.broadcast`.

    Args:
        object_list (torch.Tensor): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will be broadcast,
            but each rank must provide lists of equal sizes.
        src (int, optional): Source rank (default: ``0``)
        group (ProcessGroup, optional): The process group to work on. If ``None``,
            the default process group will be used. Default is ``None``.

    Returns:
        None:  ``object_list`` will be modified in-place and set to values of ``object_list`` from the ``src`` rank.
    """
    world = World.detect()
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.broadcast_object_list(object_list, src=src, group=group)
        # torch.distributed will replace the None's in obj_gather_list with the gathered objects on rank 0
        # or will just be None on non-rank-0
        return
    world_size = world.num_ranks
    if world_size == 1:
        return
    raise RuntimeError(
        f"The world_size({world_size}) > 1, but the distributed package is not "
        "available or has not been initialized. Please check you have initialized "
        "the distributed runtime and that PyTorch has been built with distributed "
        "support. If calling this function outside Trainer, please ensure that "
        "`composer.utils.dist.initialize_dist` has been called first.",
    )
