"""
Math Helper Functions

Adopt from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/math.py
"""

import itertools
import math
from dataclasses import dataclass
from typing import Literal, Tuple
from jaxtyping import Bool, Float

import torch
from torch import Tensor


def intersect_aabb(
    origins: torch.Tensor,
    directions: torch.Tensor,
    aabb: torch.Tensor,
    min_near: float = 0.0,
    max_bound: float = 1e10,
    invalid_value: float = 1e10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of ray intersection with AABB box

    Args:
        origins: [N,3] tensor of 3d positions
        directions: [N,3] tensor of normalized directions
        aabb: [6] array of aabb box in the form of [x_min, y_min, z_min, x_max, y_max, z_max]
        min_near: minimum value of t_min and t_max
        max_bound: Maximum value of t_max
        invalid_value: Value to return in case of no intersection

    Returns:
        t_min, t_max - two tensors of shapes N representing distance of intersection from the origin.
    """

    tx_min = (aabb[:3] - origins) / directions
    tx_max = (aabb[3:] - origins) / directions

    t_min = torch.stack((tx_min, tx_max)).amin(dim=0)
    t_max = torch.stack((tx_min, tx_max)).amax(dim=0)

    t_min = t_min.amax(dim=-1)
    t_max = t_max.amin(dim=-1)

    t_min = torch.clamp(t_min, min=min_near, max=max_bound)
    t_max = torch.clamp(t_max, min=min_near, max=max_bound)

    cond = t_max <= t_min
    t_min = torch.where(cond, invalid_value, t_min)
    t_max = torch.where(cond, invalid_value, t_max)

    return t_min, t_max


def safe_normalize(
    vectors: Float[Tensor, "*batch_dim N"],
    eps: float = 1e-10,
) -> Float[Tensor, "*batch_dim N"]:
    """Normalizes vectors.

    Args:
        vectors: Vectors to normalize.
        eps: Epsilon value to avoid division by zero.

    Returns:
        Normalized vectors.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + eps)
