import functools
import pickle
from typing import Any
from typing import Union

import loguru
import numpy as np
import torch
import torch.distributed as dist


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


class MainProcessLogger:
    def log(self, level: Union[str, int], message: str, *args, **kwargs) -> None:
        if get_rank() == 0:
            loguru.logger.log(level, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        self.log("DEBUG", message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        self.log("INFO", message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self.log("WARNING", message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        self.log("ERROR", message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        self.log("CRITICAL", message, *args, **kwargs)

    def exception(self, message: Any) -> None:
        if get_rank() == 0:
            loguru.logger.exception(message)


logger = MainProcessLogger()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger.warning(
            f"Rank {get_rank()} trying to all-gather {len(buffer) / (1024**3):.2f} GB of data on device {device}"
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, "distrbuted_utils.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    This is a memoised operation. Further calls does not result in new groups being
    created.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def to_device(b, device):
    if isinstance(b, dict):
        return {k: to_device(v, device) for k, v in b.items()}
    elif isinstance(b, list):
        return [to_device(v, device) for v in b]
    elif isinstance(b, tuple):
        return tuple(to_device(v, device) for v in b)
    elif isinstance(b, torch.Tensor):
        return b.to(device, non_blocking=True)
    return b
