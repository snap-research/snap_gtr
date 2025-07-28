from dataclasses import dataclass
from loguru import logger
import functools

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.typing import *
from utils.config_utils import PrintableConfig, to_immutable_dict

from .layers import BasicTransformerBlock, CustomUpsample


@dataclass
class TransformerV2Config(PrintableConfig):
    """Pure transformer"""
    plane_resolution: int = 32
    """plane resolution"""
    model_channels: int = 1024
    """Number of channels of plane positional embedding"""
    context_dim: Optional[int] = None
    """Number of channels of condition. If None, change to self attention"""
    out_channels: int = 80
    """Number of channels of output"""
    num_layers: int = 12
    """Number of attention layers"""
    num_heads: int = 16
    """Number of heads of attention layer"""
    num_head_channels: int = 32
    """Number of head channel of attention layer"""
    dropout: float = 0.0
    """Dropout rate"""
    gradient_checkpointing: bool = True
    """If True, apply gradient checkpoint"""
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = to_immutable_dict(
        {"use_reentrant": False}
    )
    """Gradient checkpointing kwargs"""
    init_scale: float = 0.1
    """Scale of initial values of planes embedding"""
    gated_feedforward: bool = True
    """Set true to use gated feedforward"""
    upsample_plane: bool = False
    """Set true to upsample plane to x2 resolution"""
    upsample_conv_transpose: bool = False
    """Set true to apply convTranspose2D to upsample"""


class TransformerV2(nn.Module):
    def __init__(
        self,
        cfg: TransformerV2Config,
    ):
        super().__init__()
        plane_resolution = cfg.plane_resolution
        model_channels = cfg.model_channels
        context_dim = cfg.context_dim
        num_heads = cfg.num_heads
        num_head_channels = cfg.num_head_channels
        num_layers = cfg.num_layers
        out_channels = cfg.out_channels
        dropout = cfg.dropout
        gated_feedforward = cfg.gated_feedforward
        initial_scale = cfg.init_scale

        # setup gradient checkpointing
        self.gradient_checkpointing = cfg.gradient_checkpointing
        if self.gradient_checkpointing:
            gradient_checkpointing_kwargs = cfg.gradient_checkpointing_kwargs
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            self._gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        self.planes = torch.nn.Parameter(
            initial_scale * torch.randn((1, 3, model_channels, plane_resolution, plane_resolution))
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

        if not cfg.upsample_plane:
            self.out = nn.Sequential(
                nn.Conv2d(model_channels, out_channels, 3, padding=1),
            )
        else:
            self.out = CustomUpsample(
                model_channels,
                out_channels,
                use_conv_transpose=cfg.upsample_conv_transpose,
            )

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
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
        # TODO: refactor num_parameters to be generic

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []
        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                total_numel.append(param.numel())

        return sum(total_numel)

    def forward(self, context):
        """

        Args:
            context: torch.Tensor, (batch_size, seq_len, channel)

        Returns:
            triplane_code: torch.Tensor, (batch_size, num_plane, channel, height, width)
        """
        num_scenes = context.shape[0]

        x = self.planes.expand(num_scenes, -1, -1, -1, -1)
        num_scenes, num_planes, n_channels, h, w = x.size()

        x = x.permute(0, 1, 3, 4, 2).reshape(num_scenes, num_planes*h*w, n_channels).contiguous()

        for i, layer_module in enumerate(self.attn_layers):
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    x,
                    context,
                )
            else:
                x = layer_module(x, context=context)

        x = x.view(num_scenes, num_planes, h, w, n_channels).permute(0, 1, 4, 2, 3).contiguous()

        x = self.out(x.view(num_scenes*num_planes, n_channels, h, w))
        x = x.view(num_scenes, num_planes, *x.size()[1:])

        return x



