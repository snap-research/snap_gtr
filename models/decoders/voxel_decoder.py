"""
Voxel decoder implementation
"""
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fields import BaseField
from utils.typing import *


class VoxelDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[str] = 'relu',
        sigma_activation: Optional[str] = 'softplus',
        interp_mode: Optional[str] = 'bilinear',
        sigmoid_saturation: Optional[float] = 0.001,
    ) -> None:
        super().__init__()
        self.interp_mode = interp_mode
        self.field = BaseField(
            in_dim=in_dim,
            num_layers=num_layers,
            layer_width=layer_width,
            out_dim=out_dim,
            skip_connections=skip_connections,
            activation=activation,
            sigma_activation=sigma_activation,
            sigmoid_saturation=sigmoid_saturation,
        )

    def point_decode(self, xyzs, code, density_only=False):
        """ Convert latent feature to RGB + density.
        Args:
            xyzs: Shape (num_scenes, (num_points_per_scene, 3))
            code: Shape (num_scenes, 3, n_channels, h, w)

        Returns:
            sigmas: Shape (num_scenes, (num_points_per_scene, 1))
            rgbs: Shape (num_scenes, (num_points_per_scene, 3))
        """
        # TODo: check grid_sample
        num_scenes, n_channels, d, h, w = code.size()
        _, num_points, _ = xyzs.shape
        point_code = F.grid_sample(
            code,
            xyzs.reshape(num_scenes, 1, 1, num_points, 3),
            mode=self.interp_mode, padding_mode='border', align_corners=False
        ).reshape(num_scenes, -1, num_points)
        point_code = point_code.permute(0, 2, 1).reshape(num_scenes * num_points, -1).contiguous()

        sigmas, rgbs = self.field(point_code, density_only=density_only)
        sigmas = sigmas.reshape(num_scenes, num_points, 1)
        if rgbs is not None:
            rgbs = rgbs.reshape(num_scenes, num_points, 3)

        return sigmas, rgbs

    def point_density_code(self, xyzs, code):
        sigmas, _ = self.point_decode(
            xyzs, code, density_only=True)
        return sigmas

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        num_scenes, num_chn, d, h, w = code.size()
        assert d == h and d == w
        # visualize center voxel
        code_viz = code.detach().cpu()
        xy = code_viz[:, :, d // 2, :, :]
        xz = code_viz[:, :, :, h // 2, :]
        yz = code_viz[:, :, :, :, w // 2]
        code_viz = torch.stack([xy, xz, yz], dim=1)  # (num_scenes, num_planes, num_channels, h, w)
        code_viz = code_viz.numpy()
        # num_scenes, _, num_chn, h, w = code_viz.shape
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 3 * h, num_chn * w)
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            out_file = f"{viz_dir}/scene_{scene_name_single}.jpg"
            plt.imsave(
                out_file,
                code_viz_single,
                vmin=code_range[0],
                vmax=code_range[1]
            )





