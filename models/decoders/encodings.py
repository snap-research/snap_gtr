from dataclasses import dataclass

import torch

from utils.typing import *
from utils.config_utils import PrintableConfig


@dataclass
class TriPlaneEncodingConfig(PrintableConfig):
    resolution: int = 32
    """plane resolution"""
    num_channels: int = 64
    """Number of feature channels of plane"""
    num_scenes: int = 1
    """Number of scenes"""
    init_scale: float = 0.1
    """Scale of inital values of planes"""


class TriPlaneEncoding(torch.nn.Module):
    """Learned triplane encoding"""
    def __init__(
        self,
        cfg: TriPlaneEncodingConfig
    ) -> None:
        super().__init__()

        self.resolution = cfg.resolution
        self.num_components = cfg.num_channels
        self.num_scenes = cfg.num_scenes
        self.init_scale = cfg.init_scale

        self.planes = torch.nn.Parameter(
            self.init_scale * torch.randn((self.num_scenes, 3, self.num_components, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        return self.num_components

    def get_code(self):
        return self.planes


@dataclass
class VoxelEncodingConfig(PrintableConfig):
    resolution: int = 32
    """plane resolution"""
    num_channels: int = 64
    """Number of feature channels of plane"""
    num_scenes: int = 1
    """Number of scenes"""
    init_scale: float = 0.1
    """Scale of inital values of planes"""


class VoxelEncoding(torch.nn.Module):
    """Learned triplane encoding"""
    def __init__(
        self,
        cfg: VoxelEncodingConfig,
    ) -> None:
        super().__init__()
        self.resolution = cfg.resolution
        self.num_components = cfg.num_channels
        self.num_scenes = cfg.num_scenes
        self.init_scale = cfg.init_scale

        self.voxels = torch.nn.Parameter(
            self.init_scale * torch.randn((self.num_scenes, self.num_components, self.resolution, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        return self.num_components

    def get_code(self):
        return self.voxels
