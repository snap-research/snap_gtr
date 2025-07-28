# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common Colors"""
from typing import Union
from typing import TypeVar
import numpy as np

import torch
from jaxtyping import Float
from torch import Tensor

WHITE = torch.tensor([1.0, 1.0, 1.0])
BLACK = torch.tensor([0.0, 0.0, 0.0])
RED = torch.tensor([1.0, 0.0, 0.0])
GREEN = torch.tensor([0.0, 1.0, 0.0])
BLUE = torch.tensor([0.0, 0.0, 1.0])

COLORS_DICT = {
    "white": WHITE,
    "black": BLACK,
    "red": RED,
    "green": GREEN,
    "blue": BLUE,
}


def get_color(color: Union[str, list]) -> Float[Tensor, "3"]:
    """
    Args:
        Color as a string or a rgb list

    Returns:
        Parsed color
    """
    if isinstance(color, str):
        color = color.lower()
        if color not in COLORS_DICT:
            raise ValueError(f"{color} is not a valid preset color")
        return COLORS_DICT[color]
    if isinstance(color, list):
        if len(color) != 3:
            raise ValueError(f"Color should be 3 values (RGB) instead got {color}")
        return torch.tensor(color)

    raise ValueError(f"Color should be an RGB list or string, instead got {type(color)}")


def get_color_numpy(color: Union[str, list]):
    color_tensor = get_color(color)
    return color_tensor.detach().cpu().numpy()


def _check_unit_scale(tensor: Union[np.ndarray, torch.Tensor]):
    if tensor.min() < 0 or tensor.max() > 1:
        raise ValueError("Input tensor should be unit scaled")


ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, torch.Tensor)


def linear_to_nonlinear_srgb(colors: ArrayOrTensor, *, check_unit_scale: bool = True) -> ArrayOrTensor:
    """Converts colors from linear to non-linear sRGB space.
    See https://observablehq.com/@sebastien/srgb-rgb-gamma.
    Args:
        colors (ArrayOrTensor): Linear sRGB colors tensor.
        check_unit_scale (bool): Raise exception if any value in colors is out of [0, 1] range.
    Returns:
        ArrayOrTensor: Non-linear sRGB colors tensor of the same type as input.
    """
    if check_unit_scale:
        _check_unit_scale(colors)
    mask = colors < 0.0031308
    return mask * colors * 12.92 + ~mask * (1.055 * (colors ** (1 / 2.4)) - 0.055)


def nonlinear_to_linear_srgb(colors: ArrayOrTensor, *, check_unit_scale: bool = True) -> ArrayOrTensor:
    """Converts colors from non-linear to linear sRGB space.
    See https://observablehq.com/@sebastien/srgb-rgb-gamma.
    Args:
        colors (ArrayOrTensor): Non-linear sRGB colors tensor.
        check_unit_scale (bool): Raise exception if any value in colors is out of [0, 1] range.
    Returns:
        ArrayOrTensor: Linear sRGB colors tensor of the same type as input.
    """
    if check_unit_scale:
        _check_unit_scale(colors)
    mask = colors < (0.0031308 * 12.92)
    return mask * colors / 12.92 + ~mask * (((colors + 0.055) / 1.055) ** 2.4)