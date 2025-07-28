from abc import abstractmethod
from dataclasses import dataclass
from jaxtyping import Float, Int, Shaped
from typing import Optional, Callable, Tuple, List
from loguru import logger

import torch
from torch import nn


@dataclass
class RaySamples:
    """xyz coordinate for ray origin."""
    origins: torch.Tensor  # [bs:..., 3]
    """Direction of ray."""
    directions: torch.Tensor  # [bs:..., 3]
    """Where the frustum starts along a ray."""
    starts: torch.Tensor  # [bs:..., 1]
    """Where the frustum ends along a ray."""
    ends: torch.Tensor  # [bs:..., 1]
    """"width" of each sample."""
    deltas: Optional[torch.Tensor] = None  # [bs, ...?, 1]

    def print_info(self):
        logger.debug(
            f"ray_samples: \n"
            f"\t origins: {self.origins.shape} \n"
            f"\t directions: {self.directions.shape} \n"
            f"\t starts: {self.starts.shape} \n"
            f"\t ends: {self.ends.shape} \n"
            f"\t deltas: {self.deltas.shape} \n"
        )

    def get_positions(self) -> torch.Tensor:
        """Calulates "center" position of frustum. Not weighted by mass.
        Returns:
            xyz positions (..., 3).
        """
        return self.origins + self.directions * (self.starts + self.ends) / 2  # world space

    def get_weights(self, densities: Float[torch.Tensor, "*batch num_samples 1"]) -> Float[torch.Tensor, "*batch num_samples 1"]:
        """Return weights based on predicted densities
        Args:
            densities: Predicted densities for samples along ray (..., num_samples, 1)
        Returns:
            Weights for each sample  (..., num_samples, 1)
        """
        delta_mask = self.deltas > 0
        deltas = self.deltas[delta_mask]

        delta_density = torch.zeros_like(densities)
        delta_density[delta_mask] = deltas * densities[delta_mask]
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:-2], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights


@dataclass
class RayBundle:
    """A bundle of ray parameters."""

    """Ray origins (XYZ)"""
    origins: torch.Tensor  # [..., 3]
    """Unit ray direction vector"""
    directions: torch.Tensor  # [..., 3]
    """Distance along ray to start sampling"""
    nears: Optional[torch.Tensor] = None  # [..., 1]
    """Rays Distance along ray to stop sampling"""
    fars: Optional[torch.Tensor] = None  # [..., 1]

    def __len__(self):
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def get_ray_samples(
        self,
        bin_starts: torch.Tensor,
        bin_ends: torch.Tensor,
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction.
        Args:
            bin_starts: Distance from origin to start of bin.
                TensorType["bs":..., "num_samples", 1]
            bin_ends: Distance from origin to end of bin.
        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        return RaySamples(
            origins=self.origins[..., None, :],  # [..., 1, 3]
            directions=self.directions[..., None, :],  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1]  world
            ends=bin_ends,  # [..., num_samples, 1]      world
            deltas=deltas,  # [..., num_samples, 1]  world coo
        )


class Sampler(nn.Module):
    """Generate Samples
    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def generate_ray_samples(self) -> RaySamples:
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs) -> RaySamples:
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)


class SpacedSampler(Sampler):
    """Sample points according to a function.
    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    # noinspection PyMethodOverriding
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples accoring to spacing function.
        Args:
            ray_bundle: Ray-origins, directions, etc.
            num_samples: Number of samples per ray
        Returns:
            Positions and deltas for samples along a ray
        """
        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        rays_o, ray_nears, ray_fars = ray_bundle.origins, ray_bundle.nears, ray_bundle.fars
        assert ray_nears.ndim == 3 and ray_fars.ndim == 3
        num_scenes, num_scene_rays, _ = rays_o.shape

        # sample bins for each ray
        num_rays = num_scenes * num_scene_rays
        device = rays_o.device
        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(device)[None, ...]  # [1, num_samples+1]
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand
        else:
            bins = bins.repeat(num_rays, 1)
        bins = bins.view(num_scenes, num_scene_rays, num_samples+1)

        # s_near, s_far in [0, 1], euclidean in world
        s_near, s_far = (self.spacing_fn(x) for x in (ray_nears, ray_fars))
        spacing_to_euclidean_fn = lambda x: self.spacing_fn_inv(x * s_far + (1 - x) * s_near)
        assert bins.ndim == s_near.ndim
        euclidean_bins = spacing_to_euclidean_fn(bins)  # [..., num_samples+1]

        bin_starts = euclidean_bins[..., :-1, None]
        bin_ends = euclidean_bins[..., 1:, None]
        return ray_bundle.get_ray_samples(
            bin_starts=bin_starts,  # world [near, far]
            bin_ends=bin_ends,     # world [near, far]
        )


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray
    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )
