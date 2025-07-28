"""
Collection of renderers
"""
from typing import Generator, Literal, Optional, Tuple, Union
from jaxtyping import Float, Int
from loguru import logger

import torch
from torch import Tensor, nn

from utils.math import safe_normalize
from utils.ray_samples import RaySamples
from utils.distributed import get_rank


class RGBRenderer(nn.Module):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self) -> None:
        super().__init__()

    def combine_rgb(
        self,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image.
        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.

        Returns:
            Outputs rgb values.
        """
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        assert isinstance(background_color, torch.Tensor)
        background_color = background_color.expand(comp_rgb.shape).to(comp_rgb.device)
        comp_rgb = comp_rgb + background_color * (1.0 - accumulated_weight)
        return comp_rgb

    def forward(
        self,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: Float[Tensor, "*bs num_samples 3"],
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color for each ray

        Returns:
            Outputs of rgb values.
        """
        if not self.training:
            rgb = torch.nan_to_num(rgb)

        rgb = self.combine_rgb(
            rgb, weights, background_color=background_color
        )

        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb


class AccumulationRenderer(nn.Module):
    """Accumulated value along a ray."""

    @classmethod
    def forward(
        cls,
        weights: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs 1"]:
        """Composite samples along ray and calculate accumulation.

        Args:
            weights: Weights for each sample

        Returns:
            Outputs of accumulated values.
        """

        accumulation = torch.sum(weights, dim=-2)
        return accumulation


class DepthRenderer(nn.Module):
    """Calculate depth along ray.
    Depth Method:
        - median: Depth is set to the distance where the accumulated weight reaches 0.5.
        - expected: Expected depth along ray. Same procedure as rendering rgb, but with depth.

    Args:
        method: Depth calculation method.
    """
    def __init__(self, method: Literal["median", "expected"] = "median") -> None:
        super().__init__()
        self.method = method

    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
    ) -> Float[Tensor, "*batch 1"]:
        """Composite samples along ray and calculate depths.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.

        Returns:
            Outputs of depth values.
        """
        if self.method == "median":
            steps = (ray_samples.starts + ray_samples.ends) / 2
            cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
            split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
            median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
            median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
            median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [..., 1]
            return median_depth

        elif self.method == 'expected':
            eps = 1e-10
            steps = (ray_samples.starts + ray_samples.ends) / 2
            depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
            depth = torch.clip(depth, steps.min(), steps.max())

            return depth

        else:
            raise NotImplementedError(f"Method {self.method} not implemented")


class NormalsRenderer(nn.Module):
    """Calculate normals along the ray."""

    @classmethod
    def forward(
        cls,
        normals: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        normalize: bool = True,
    ) -> Float[Tensor, "*bs 3"]:
        """Calculate normals along the ray.

        Args:
            normals: Normals for each sample.
            weights: Weights of each sample.
            normalize: Normalize normals.
        """
        n = torch.sum(weights * normals, dim=-2)
        if normalize:
            n = safe_normalize(n)
        return n


class UncertaintyRenderer(nn.Module):
    """Calculate uncertainty along the ray."""

    @classmethod
    def forward(
        cls, betas: Float[Tensor, "*bs num_samples 1"], weights: Float[Tensor, "*bs num_samples 1"]
    ) -> Float[Tensor, "*bs 1"]:
        """Calculate uncertainty along the ray.

        Args:
            betas: Uncertainty betas for each sample.
            weights: Weights of each sample.

        Returns:
            Rendering of uncertainty.
        """
        uncertainty = torch.sum(weights * betas, dim=-2)
        return uncertainty
