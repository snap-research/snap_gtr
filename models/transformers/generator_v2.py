"""
Take image as input and output triplane embeddings
"""
from dataclasses import dataclass
from loguru import logger
import functools
from einops import rearrange

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.typing import *
from utils.config_utils import PrintableConfig, to_immutable_dict

from .layers import BasicTransformerBlock, PSUpsampler


@dataclass
class ModelConfig(PrintableConfig):
    """Pure transformer"""
    plane_resolution: int = 32
    """initial triplane resolution"""
    out_channels: int = 80
    """Number of channels of output"""
    in_channels: int = 9
    """Number of channels of input images"""
    patch_size: int = 16
    """Image patch size"""
    model_channels: int = 1024
    """Number of channels of plane positional embedding"""
    num_layers: int = 16
    """Number of attention layers"""
    num_heads: int = 16
    """Number of heads of attention layer"""
    num_head_channels: int = 64
    """Number of head channel of attention layer"""
    dropout: float = 0.0
    """Dropout rate"""
    gated_feedforward: bool = True
    """Set true to use gated feedforward"""
    gradient_checkpointing: bool = True
    """If True, apply gradient checkpoint"""
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = to_immutable_dict(
        {"use_reentrant": False}
    )
    """Gradient checkpointing kwargs"""
    init_scale: float = 0.1
    """Scale of initial values of planes embedding"""
    upsample_factor: int = 8
    """TODO: ratio to upsample plane embedding"""
    split_triplane_view: bool = True
    """Set true to treat triplane embedding and views embedding separately"""


class GeneratorV2(nn.Module):
    """Apply a sequence of (cross_attn + self_attn) to (triplane embeddings, image_embeddings) input"""
    def __init__(
        self,
        cfg: ModelConfig
    ):
        super().__init__()
        in_channels = cfg.in_channels
        model_channels = cfg.model_channels
        num_layers = cfg.num_layers
        num_heads = cfg.num_heads
        num_head_channels = cfg.num_head_channels
        self.split_triplane_view_embeddings = cfg.split_triplane_view
        context_dim = model_channels if self.split_triplane_view_embeddings else None
        dropout = cfg.dropout
        gated_feedforward = cfg.gated_feedforward
        plane_resolution = cfg.plane_resolution
        initial_scale = cfg.init_scale
        out_channels = cfg.out_channels
        upsample_factor = cfg.upsample_factor
        patch_size = cfg.patch_size

        # models
        self.planes = torch.nn.Parameter(
            initial_scale * torch.randn((1, 3, model_channels, plane_resolution, plane_resolution))
        )

        self.conv_proj = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=model_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        layers = []
        for i in range(num_layers):
            layers.append(BasicTransformerBlock(
                dim=model_channels,
                n_heads=num_heads,
                d_head=num_head_channels,
                context_dim=context_dim,
                dropout=dropout,
                gated_ff=gated_feedforward
            ))
        self.attn_layers = nn.ModuleList(layers)

        self.out = PSUpsampler(
            model_channels,
            out_channels,
            scale_factor=upsample_factor,
        )

        # others
        self.gradient_checkpointing = cfg.gradient_checkpointing
        if self.gradient_checkpointing:
            gradient_checkpointing_kwargs = cfg.gradient_checkpointing_kwargs
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            self._gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing

    def forward(self, images):
        """
        Args:
            images: torch.Tensor, (batch_size, num_views, channel, height, width)

        Returns:
            triplane_code: torch.Tensor, (batch_size, num_plane, channel, height, width)
        """
        assert images.ndim == 5
        num_scenes, num_views = images.shape[:2]
        context = rearrange(images, 'b v c h w -> (b v) c h w')
        context = self.conv_proj(context)
        context = context.view(num_scenes, num_views, *context.size()[1:])
        context = rearrange(context, 'b v c h w -> b (v h w) c').contiguous()

        x = self.planes.expand(num_scenes, -1, -1, -1, -1)
        num_scenes, num_planes, n_channels, h, w = x.size()
        x = x.permute(0, 1, 3, 4, 2).reshape(num_scenes, num_planes*h*w, n_channels).contiguous()

        if not self.split_triplane_view_embeddings:
            x = torch.cat((x, context), dim=1)  #
            context = None
        for i, layer_module in enumerate(self.attn_layers):
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    x,
                    context,
                )
            else:
                x = layer_module(x, context=context)
        if not self.split_triplane_view_embeddings:
            x = x[:, :num_planes*h*w, :]

        x = x.view(num_scenes, num_planes, h, w, n_channels).permute(0, 1, 4, 2, 3).contiguous()
        x = self.out(x.view(num_scenes*num_planes, n_channels, h, w))
        x = x.view(num_scenes, num_planes, *x.size()[1:])

        return x

