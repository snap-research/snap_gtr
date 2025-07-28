from dataclasses import dataclass
from loguru import logger

import torch
import torch.nn as nn

from .ldm_modules import (
    EmbedSequential,
    SpatialTransformer,
    SpatialTransformer3D,
    Downsample,
    Upsample,
    BasicTransformerBlockV2,
)
from .ldm_utils import (
    conv_nd,
    zero_module,
    normalization,
)

from utils.typing import *
from utils.config_utils import PrintableConfig


class ResBlockV2(nn.Module):
    def __init__(
        self,
        in_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, in_channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


@dataclass
class TransformerUNetV1Config(PrintableConfig):
    """Transformer block similar to diffusion unet"""
    in_channels: int
    """Number of channels of input"""
    model_channels: int
    """Number of channels of model intermediate embedding"""
    out_channels: int
    """Number of channels of output"""
    context_dim: int
    """Number of channels of condition"""
    num_res_blocks: int
    """Number of blocks (ConvBlock + [option]attention) per level"""
    attention_levels: Tuple[int, ...] = (16, 8, 4)
    """Specifies a list of levels to perform attention"""
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4, 4)
    """Specifies a list of intermediate channel multiplication for each level"""
    disable_updown: Optional[Tuple[bool, ...]] = (False, False, False, False)
    """Specifies whether to skip updown at level"""
    conv_resample: bool = True
    """If true, apply convolution for upsample"""
    dims: int = 2
    """Specify input dimension"""
    num_heads: int = 4
    """Number of heads of attention layer"""
    num_head_channels: int = 32
    """Number of head channel of attention layer"""
    transformer_depth: int = 1
    """Number of basic attention layers per transformer block"""
    dropout: float = 0.0
    """Dropout """
    use_checkpoint: bool = True
    """If True, apply checkpoint"""


class TransformerUNetV1(nn.Module):
    def __init__(
        self,
        cfg: TransformerUNetV1Config,
    ):
        super().__init__()
        in_channels = cfg.in_channels
        model_channels = cfg.model_channels
        out_channels = cfg.out_channels
        context_dim = cfg.context_dim
        num_res_blocks = cfg.num_res_blocks
        attention_levels = cfg.attention_levels
        dropout = cfg.dropout
        channel_mult = cfg.channel_mult
        conv_resample = cfg.conv_resample
        use_checkpoint = cfg.use_checkpoint
        num_heads = cfg.num_heads
        num_head_channels = cfg.num_head_channels
        transformer_depth = cfg.transformer_depth
        dims = cfg.dims
        disable_updown = cfg.disable_updown
        if disable_updown is None:
            disable_updown = [False] * (len(channel_mult)-1)  # Default is downsample at each level
        assert len(disable_updown) == (len(channel_mult) - 1)

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        spatial_transformer_module = SpatialTransformer if dims == 2 else SpatialTransformer3D

        self.input_blocks = nn.ModuleList(
            [
                EmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlockV2(
                        in_channels=ch,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                    ),
                ]
                ch = mult * model_channels
                if ds in attention_levels:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        spatial_transformer_module(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            checkpoint=use_checkpoint,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                # downsample from level to (level + 1)
                if disable_updown[level]:
                    # replace downsample layer with identity layer
                    self.input_blocks.append(
                        EmbedSequential(
                            nn.Identity()
                        )
                    )
                    ds *= 1
                    input_block_chans.append(ch)
                    self._feature_size += ch
                else:
                    out_ch = ch
                    self.input_blocks.append(
                        EmbedSequential(
                            Downsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )
                        )
                    )
                    ch = out_ch
                    input_block_chans.append(ch)
                    ds *= 2
                    self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = EmbedSequential(
            ResBlockV2(
                in_channels=ch,
                dropout=dropout,
                out_channels=ch,
                dims=dims,
            ),
            spatial_transformer_module(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                checkpoint=use_checkpoint
            ),
            ResBlockV2(
                in_channels=ch,
                dropout=dropout,
                out_channels=ch,
                dims=dims
            )
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockV2(
                        in_channels=ch + ich,
                        dropout=dropout,
                        out_channels=model_channels*mult,
                        dims=dims,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_levels:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        spatial_transformer_module(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            checkpoint=use_checkpoint,
                        )
                    )
                if level and i == num_res_blocks:
                    # reverse ordering
                    if disable_updown[level-1]:
                        layers.append(
                            EmbedSequential(
                                nn.Identity()
                            )
                        )
                    else:
                        out_ch = ch
                        layers.append(
                            Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x,
        context=None,
        concat_cond=None,
    ):
        """
        Args:
            x: torch.tensor, (B, C, ...)
            context: torch.tensor, (B, seq_len, C2), cross attention condition
            concat_cond: torch.tensor, (B, C3, ... ), spatial concat condition
        """

        h, hs = x, []
        if concat_cond is not None:
            h = torch.cat([h, concat_cond], dim=1)  # spatial concat

        # print(f"-------------------- input blocks ------------------------")
        for module in self.input_blocks:
            h = module(h, context)
            hs.append(h)
        # print(f"-------------------- middle blocks ------------------------")
        h = self.middle_block(h, context)
        # print(f"-------------------- output blocks ------------------------")
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, context)

        outputs = self.out(h)

        return outputs


@dataclass
class TransformerV1Config(PrintableConfig):
    """Pure transformer"""
    plane_resolution: int
    """plane resolution"""
    model_channels: int
    """Number of channels of model intermediate embedding"""
    out_channels: int
    """Number of channels of output"""
    num_layers: int
    """Number of attention layers"""
    context_dim: Optional[int] = None
    """Number of channels of condition. If None, change to self attention"""
    channel_mult: Optional[Tuple[int, ...]] = (1, 2, 2, 4, 4)
    """Specifies a list of intermediate channel multiplication for each level"""
    num_heads: int = 4
    """Number of heads of attention layer"""
    num_head_channels: int = 32
    """Number of head channel of attention layer"""
    dropout: float = 0.0
    """Dropout """
    use_checkpoint: bool = True
    """If True, apply checkpoint"""
    init_scale: float = 0.1
    """Scale of initial values of planes embedding"""


class TransformerV1(nn.Module):
    def __init__(
        self,
        cfg: TransformerV1Config,
    ):
        super().__init__()
        model_channels = cfg.model_channels
        context_dim = cfg.context_dim
        out_channels = cfg.out_channels
        dropout = cfg.dropout
        num_heads = cfg.num_heads
        num_head_channels = cfg.num_head_channels
        num_layers = cfg.num_layers
        plane_resolution = cfg.plane_resolution
        initial_scale = cfg.init_scale

        self.planes = torch.nn.Parameter(
            initial_scale * torch.randn((1, 3, model_channels, plane_resolution, plane_resolution))
        )

        layers = []
        for i in range(num_layers):
            layers.append(BasicTransformerBlockV2(
                dim=model_channels,
                n_heads=num_heads,
                d_head=num_head_channels,
                context_dim=context_dim,
                dropout=dropout
            ))
        self.attn_layers = nn.ModuleList(layers)

        # TODO: triplane up-sample block
        self.out = nn.Sequential(
            # normalization(model_channels),
            # nn.SiLU(),
            conv_nd(2, model_channels, out_channels, 3, padding=1),
        )

    def forward(self, context):
        """

        Args:
            x: triplane learnable embedding, (b, 3, n_channel, height, width)
            context:

        Returns:

        """
        num_scenes = context.shape[0]

        x = self.planes.expand(num_scenes, -1, -1, -1, -1)
        assert x.ndim == 5
        num_scenes, num_planes, n_channels, h, w = x.size()

        x = x.permute(0, 1, 3, 4, 2).reshape(num_scenes, -1, n_channels).contiguous()
        for i, layer in enumerate(self.attn_layers):
            x = layer(x, context=context)
        x = x.view(num_scenes, num_planes, h, w, n_channels).permute(0, 1, 4, 2, 3).contiguous()

        # upsample planes
        x = self.out(x.view(-1, n_channels, h, w)).view(num_scenes, num_planes, -1, h, w)

        return x
