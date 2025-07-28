"""
Customized dinov2
"""

from dataclasses import dataclass

import torch
from torchvision import transforms
import torch.nn as nn

from utils.typing import *
from utils.config_utils import PrintableConfig


DINOv2_REPO = "facebookresearch/dinov2"

@dataclass
class DINOv2Config(PrintableConfig):
    model_name: str = "dinov2_vitb14"
    """Model name """
    input_channels: int = 3
    """Number of channels of model input """


class DINOv2(torch.nn.Module):
    def __init__(
        self,
        cfg: DINOv2Config,
    ):
        super().__init__()
        model_name = cfg.model_name
        input_channels = cfg.input_channels

        self.in_layers = nn.Sequential()
        embed_dim = 3
        self.input_proj = nn.Conv2d(input_channels, embed_dim, kernel_size=3, padding=1)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Dino takes 224 input, proper positional encoding???
        self.preprocess = transforms.Compose(
            [
                transforms.CenterCrop(224),
            ]
        )

        self.dino = torch.hub.load(DINOv2_REPO, model_name)

    @property
    def device(self):
        """torch.device where model is placed"""
        return next(self.dino.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns image features tensor.
        For patch embedding_type, return (B, 256, C)
        """
        x = self.preprocess(x)  # center crop to (224, 224)
        x = self.input_proj(x)

        res = self.dino.forward_features(x)
        patchtokens = res['x_norm_patchtokens']
        classtokens = res['x_norm_clstoken'].unsqueeze(dim=1)
        res = torch.concat([classtokens, patchtokens], dim=1)

        return res
