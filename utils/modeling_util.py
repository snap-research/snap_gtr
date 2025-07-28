import torch

from utils.typing import *


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def scale_dict(dictionary: Dict[Any, Any], coefficients: Dict[str, float]) -> Dict[Any, Any]:
    """Scale a dictionary in-place given a coefficients dictionary.

    Args:
        dictionary: input dict to be scaled.
        coefficients: scalar dict config for holding coefficients.

    Returns:
        Input dict scaled by coefficients.
    """
    for key in dictionary:
        if key in coefficients:
            dictionary[key] *= coefficients[key]
    return dictionary


def _get_autocast_kwargs():
    gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                           "dtype": torch.get_autocast_gpu_dtype(),
                           "cache_enabled": torch.is_autocast_cache_enabled()}

    cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                           "dtype": torch.get_autocast_cpu_dtype(),
                           "cache_enabled": torch.is_autocast_cache_enabled()}

    return gpu_autocast_kwargs, cpu_autocast_kwargs


def count_model_num_parameters(model, only_trainable: bool = False, exclude_embeddings: bool = False):
    """
    Get number of (optionally, trainable or non-embeddings) parameters in the module.

    Args:
        only_trainable (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of trainable parameters

        exclude_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of non-embeddings parameters

    Returns:
        `int`: The number of parameters.
    """

    if exclude_embeddings:
        embedding_param_names = [
            f"{name}.weight" for name, module_type in model.named_modules() if isinstance(module_type, torch.nn.Embedding)
        ]
        total_parameters = [
            parameter for name, parameter in model.named_parameters() if name not in embedding_param_names
        ]
    else:
        total_parameters = list(model.parameters())

    total_numel = []
    for param in total_parameters:
        if param.requires_grad or not only_trainable:
            total_numel.append(param.numel())

    return sum(total_numel)

